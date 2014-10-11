/*
 * FaceVacsEngine.hpp
 *
 *  Created on: 11.10.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FACEVACSENGINE_HPP_
#define FACEVACSENGINE_HPP_

#include "facerecognition/FaceRecord.hpp"
#include "facerecognition/frsdk.hpp"

#include "logging/LoggerFactory.hpp"

#include "frsdk/config.h"
#include "frsdk/enroll.h"
#include "frsdk/match.h"
#include "frsdk/eyes.h"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include "boost/optional.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

//#include <string>
#include <memory>
#include <fstream>

using logging::LoggerFactory;
using boost::filesystem::path;
using std::cout;
using std::endl;

namespace facerecognition {

// for the CoutEnrol
std::ostream&
operator<<(std::ostream& o, const FRsdk::Position& p)
{
	o << "[" << p.x() << ", " << p.y() << "]";
	return o;
}

/**
 * Representation for a face record. Can e.g. come from MultiPIE or from PaSC XML.
 * Next, we probably need loader (or createFrom(...)), e.g. for MultiPIE load from the filename, from PaSC from XML.
 */
// templates instead of inheritance? (compile-time fine, don't need runtime selection)
// or 'FaceVacsMatcher'
class FaceVacsEngine
{
public:
	// tempDirectory a path where temp files stored... only FIRs? More? firDir?
	FaceVacsEngine(path frsdkConfig, path tempDirectory) : tempDir(tempDirectory) {
		// Todo: Create the tempDir if it doesn't exist yet

		// initialize and resource allocation
		cfg = std::make_unique<FRsdk::Configuration>(frsdkConfig.string());
		firBuilder = std::make_unique<FRsdk::FIRBuilder>(*cfg.get());
		// initialize matching facility
		me = std::make_unique<FRsdk::FacialMatchingEngine>(*cfg.get());
		// load the fir population (gallery)
		population = std::make_unique<FRsdk::Population>(*cfg.get());

		// Add somewhere: 
		//catch (const FRsdk::FeatureDisabled& e) 
		//catch (const FRsdk::LicenseSignatureMismatch& e)
	};

	// because it often stays the same
	void enrollGallery(std::vector<facerecognition::FaceRecord> galleryRecords, path databasePath)
	{
		// We first enroll the whole gallery:
		auto logger = Loggers->getLogger("facerecognition");
		logger.info("Enroling gallery, " + std::to_string(galleryRecords.size()) + " records...");

		// create an enrollment processor
		FRsdk::Enrollment::Processor proc(*cfg);
		// Todo: FaceVACS should be able to do batch-enrolment?
		auto cnt = 0;
		for (auto& r : galleryRecords) {
			++cnt;
			std::cout << cnt << std::endl;
			// Does the FIR already exist in the temp-dir? If yes, skip FD/EyeDet!
			auto firPath = tempDir / r.dataPath;
			firPath.replace_extension(".fir");
			if (boost::filesystem::exists(firPath)) {
				continue;
			}

			boost::optional<path> fir = createFir(databasePath / r.dataPath);
			if (!fir) {
				continue;
			}
		}
		// Enrol all the FIRs now:
		for (auto& r : galleryRecords) {
			auto firPath = tempDir / r.dataPath;
			firPath.replace_extension(".fir");
			std::ifstream firIn(firPath.string(), std::ios::in | std::ios::binary);
			// Some of the FIRs might not exist due to enrolment failure
			//std::cout << firPath.string() << std::endl;
			if (firIn.is_open() && firIn.good()) {
				try {
					auto fir = firBuilder->build(firIn);
					population->append(fir, firPath.string());
					this->enrolledGalleryRecords.push_back(r); // we make a copy of the record? not sure?
				}
				catch (const FRsdk::FeatureDisabled& e) {
					std::cout << e.what() << std::endl;
				}
				catch (const FRsdk::LicenseSignatureMismatch& e) {
					std::cout << e.what() << std::endl;
				}
				catch (std::exception& e) {
					std::cout << e.what() << std::endl;
				}
			}
		}
	};

	// Make a matchAgainstGallery or matchSingle function as well
	// matches a pair of images, Cog LMs
	double match(cv::Mat first, cv::Mat second)
	{
		return 0.0;
	};
	// Always pass in images/image-filenames. The "outside" should never have to worry
	// about FIRs or other intermediate representations.
	// Eyes: Currently unused, use PaSC-tr eyes in the future
	float match(path g, std::string gallerySubjectId, cv::Mat p, std::string probeFilename, cv::Vec2f firstEye, cv::Vec2f secondEye)
	{
		auto galleryFirPath = tempDir / g.filename();
		galleryFirPath.replace_extension(".fir"); // we probably don't need this - only the basename / subject ID (if it is already enroled)

		// TODO: 2 versions of this function: path and Mat. Change the Mat version => remove imwrite.
		auto tempProbeImageFilename = tempDir / path(probeFilename).filename();
		cv::imwrite(tempProbeImageFilename.string(), p);

		boost::optional<path> probeFrameFir = createFir(tempProbeImageFilename);
		if (!probeFrameFir) {
			std::cout << "Couldn't enroll the probe - not a good frame. Return score 0.0." << std::endl;
			return 0.0; // Couldn't enroll the probe - not a good frame. Return score 0.0.
		}


		std::ifstream firStream(probeFrameFir->string(), std::ios::in | std::ios::binary);
		FRsdk::FIR fir = firBuilder->build(firStream);

		//compare() does not care about the configured number of Threads
		//for the comparison algorithm. It uses always one thrad to
		//compare all inorder to preserve the order of the scores
		//according to the order in the population (orer of adding FIRs to
		//the population)
		FRsdk::CountedPtr<FRsdk::Scores> scores = me->compare(fir, *population);

		// Find the score from "g" in the gallery
		auto subjectInGalleryIter = std::find_if(begin(enrolledGalleryRecords), end(enrolledGalleryRecords), [gallerySubjectId](const facerecognition::FaceRecord& g) { return (g.subjectId == gallerySubjectId); });
		if (subjectInGalleryIter == end(enrolledGalleryRecords)) {
			return 0.0; // We didn't find the given gallery image / subject ID in the enroled gallery. Throw?
		}
		auto subjectInGalleryIdx = std::distance(begin(enrolledGalleryRecords), subjectInGalleryIter);
		auto givenProbeVsGivenGallery = std::next(begin(*scores), subjectInGalleryIdx);
		return *givenProbeVsGivenGallery;
	};

	std::vector<float> matchAll(cv::Mat p, std::string probeFilename, cv::Vec2f firstEye, cv::Vec2f secondEye)
	{
		auto tempProbeImageFilename = tempDir / path(probeFilename).filename();
		tempProbeImageFilename.replace_extension(".fir");
		boost::optional<path> probeFrameFir = createFir(matToFRsdkImage(p), tempProbeImageFilename);
		if (!probeFrameFir) {
			std::cout << "Couldn't enroll the probe - not a good frame. Return score 0.0." << std::endl;
			return std::vector<float>{};
		}

		std::ifstream firStream(probeFrameFir->string(), std::ios::in | std::ios::binary);
		FRsdk::FIR fir = firBuilder->build(firStream);

		//compare() does not care about the configured number of Threads
		//for the comparison algorithm. It uses always one thread to
		//compare all in order to preserve the order of the scores
		//according to the order in the population (order of adding FIRs to
		//the population)
		FRsdk::CountedPtr<FRsdk::Scores> scores = me->compare(fir, *population);

		return std::vector<float>{ std::make_move_iterator(std::begin(*scores)), std::make_move_iterator(std::end(*scores)) };
	};

	// matches a pair of images, given LMs as Cog init
	// match(fn, fn, ...)
	// match(Mat, Mat, ...)

	// creates the FIR in temp-dir. Returns path to FIR if successful.
	// Todo: Directly return FIR?
	boost::optional<path> createFir(path image)
	{
		// create an enrollment processor
		FRsdk::Enrollment::Processor proc(*cfg);
		// Todo: FaceVACS should be able to do batch-enrolment?
		
		// Does the FIR already exist in the temp-dir? If yes, skip FD/EyeDet!
		auto firPath = tempDir / image.filename();
		firPath.replace_extension(".fir");
		if (boost::filesystem::exists(firPath)) {
			//throw std::runtime_error("Unexpected. Should check for that.");
			// just overwrite
		}

		return createFir(FRsdk::ImageIO::load(image.string()), firPath);
		
	};
	// Todos etc see one function above!
	// Todo: We can remove 'firPath' here once we replace EnrolCoutFeedback by a direct-FIR-return class
	boost::optional<path> createFir(FRsdk::Image image, path firPath)
	{
		// create an enrollment processor
		FRsdk::Enrollment::Processor proc(*cfg);
		// Todo: FaceVACS should be able to do batch-enrolment?

		FRsdk::SampleSet enrollmentImages;
		auto sample = FRsdk::Sample(image);
		// Once we have pasc-still-training landmarks from Ross, we can use those here. In the mean-time, use the Cog Eyefinder:
		FRsdk::Face::Finder faceFinder(*cfg);
		FRsdk::Eyes::Finder eyesFinder(*cfg);
		float mindist = 0.005f; // minEyeDist, def = 0.1; me: 0.01; philipp: 0.005
		float maxdist = 0.3f; // maxEyeDist, def = 0.4; me: 0.3; philipp: 0.4
		FRsdk::Face::LocationSet faceLocations = faceFinder.find(image, mindist, maxdist);
		if (faceLocations.empty()) {
			std::cout << "FaceFinder: No face found." << std::endl;
			return boost::none;
		}
		// We just use the first found face:
		// doing eyes finding
		FRsdk::Eyes::LocationSet eyesLocations = eyesFinder.find(image, *faceLocations.begin());
		if (eyesLocations.empty()) {
			std::cout << "EyeFinder: No eyes found." << std::endl;
			return boost::none;
		}
		auto foundEyes = eyesLocations.begin(); // We just use the first found eyes
		auto firstEye = FRsdk::Position(foundEyes->first.x(), foundEyes->first.y()); // first eye (image left-most), second eye (image right-most)
		auto secondEye = FRsdk::Position(foundEyes->second.x(), foundEyes->second.y());
		sample.annotate(FRsdk::Eyes::Location(firstEye, secondEye, foundEyes->firstConfidence, foundEyes->secondConfidence));
		enrollmentImages.push_back(sample);
		// create the needed interaction instances
		FRsdk::Enrollment::Feedback feedback(new EnrolCoutFeedback(firPath.string()));
		// Note: We could do a SignalBasedFeedback, that sends a signal. We could then here wait for the signal and process stuff, i.e. get the FIR directly.
		// do the enrollment
		proc.process(enrollmentImages.begin(), enrollmentImages.end(), feedback);
		// Better solution would be to change the callback into some event-based stuff
		if (boost::filesystem::exists(firPath)) {
			return firPath;
		}
		else {
			return boost::none;
		}
	};

private:
	path tempDir;
	std::unique_ptr<FRsdk::Configuration> cfg; // static? (recommendation by fvsdk doc)
	std::unique_ptr<FRsdk::FIRBuilder> firBuilder;
	std::unique_ptr<FRsdk::FacialMatchingEngine> me;
	std::vector<facerecognition::FaceRecord> enrolledGalleryRecords; // We keep this for the subject IDs
	std::unique_ptr<FRsdk::Population> population; // enrolled gallery FIRs, same order

	class EnrolCoutFeedback : public FRsdk::Enrollment::FeedbackBody
	{
	public:
		EnrolCoutFeedback(const std::string& firFilename)
			: firFN(firFilename), firvalid(false) { }
		~EnrolCoutFeedback() {}

		void start() {
			firvalid = false;
			std::cout << "start" << std::endl;
		}

		void processingImage(const FRsdk::Image& img)
		{
			std::cout << "processing image[" << img.name() << "]" << std::endl;
		}

		void eyesFound(const FRsdk::Eyes::Location& eyeLoc)
		{
			std::cout << "found eyes at [" << eyeLoc.first
				<< " " << eyeLoc.second << "; confidences: "
				<< eyeLoc.firstConfidence << " "
				<< eyeLoc.secondConfidence << "]" << std::endl;
		}

		void eyesNotFound()
		{
			std::cout << "eyes not found" << std::endl;
		}

		void sampleQualityTooLow() {
			std::cout << "sampleQualityTooLow" << std::endl;
		}


		void sampleQuality(const float& f) {
			std::cout << "Sample Quality: " << f << std::endl;
		}

		void success(const FRsdk::FIR& fir_)
		{
			fir = new FRsdk::FIR(fir_);
			std::cout
				<< "successful enrollment";
			if (firFN != std::string("")) {

				std::cout << " FIR[filename,id,size] = [\""
					<< firFN.c_str() << "\",\"" << (fir->version()).c_str() << "\","
					<< fir->size() << "]";
				// write the fir
				std::ofstream firOut(firFN.c_str(),
					std::ios::binary | std::ios::out | std::ios::trunc);
				firOut << *fir;
			}
			firvalid = true;
			std::cout << std::endl;
		}

		void failure() { std::cout << "failure" << std::endl; }

		void end() { std::cout << "end" << std::endl; }

		const FRsdk::FIR& getFir() const {
			// call only if success() has been invoked    
			if (!firvalid)
				throw InvalidFIRAccessError();

			return *fir;
		}

		bool firValid() const {
			return firvalid;
		}

	private:
		FRsdk::CountedPtr<FRsdk::FIR> fir;
		std::string firFN;
		bool firvalid;
	};

	class InvalidFIRAccessError : public std::exception
	{
	public:
		InvalidFIRAccessError() throw() :
			msg("Trying to access invalid FIR") {}
		~InvalidFIRAccessError() throw() { }
		const char* what() const throw() { return msg.c_str(); }
	private:
		std::string msg;
	};

};

} /* namespace facerecognition */
#endif /* FACEVACSENGINE_HPP_ */
