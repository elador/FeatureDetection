/*
 * FaceVacsEngine.cpp
 *
 *  Created on: 14.10.2014
 *      Author: Patrik Huber
 */

#include "facerecognition/FaceVacsEngine.hpp"

#include "facerecognition/frsdk.hpp"
#include "facerecognition/EnrolCoutFeedback.hpp"
#include "facerecognition/ThreadPool.hpp"

#include "logging/LoggerFactory.hpp"

#include "frsdk/config.h"
#include "frsdk/enroll.h"
#include "frsdk/match.h"
#include "frsdk/fir.h"
#include "frsdk/eyes.h"

#include "opencv2/highgui/highgui.hpp"

using logging::LoggerFactory;
using boost::filesystem::path;
using std::cout;
using std::endl;

namespace facerecognition {

FaceVacsEngine::FaceVacsEngine(path frsdkConfig, path tempDirectory) : tempDir(tempDirectory)
{
	// Todo: Create the tempDir if it doesn't exist yet

	// initialize and resource allocation
	//cfg = std::make_unique<FRsdk::Configuration>(frsdkConfig.string());
	cfg = std::unique_ptr<FRsdk::Configuration>(new FRsdk::Configuration(frsdkConfig.string()));
	//firBuilder = std::make_unique<FRsdk::FIRBuilder>(*cfg.get());
	firBuilder = std::unique_ptr<FRsdk::FIRBuilder>(new FRsdk::FIRBuilder(*cfg.get()));
	// initialize matching facility
	//me = std::make_unique<FRsdk::FacialMatchingEngine>(*cfg.get());
	me = std::unique_ptr<FRsdk::FacialMatchingEngine>(new FRsdk::FacialMatchingEngine(*cfg.get()));
	// load the fir population (gallery)
	//population = std::make_unique<FRsdk::Population>(*cfg.get());
	population = std::unique_ptr<FRsdk::Population>(new FRsdk::Population(*cfg.get()));

	// Add somewhere: 
	//catch (const FRsdk::FeatureDisabled& e) 
	//catch (const FRsdk::LicenseSignatureMismatch& e)
}

FaceVacsEngine::~FaceVacsEngine()
{
	// Not sure if we have to call the destructor of 'cfg' or something.
	// Without this destructor, we get an error because 'Configuration' is only forward-declared.
	// cfg->~Configuration();
	// cfg.release(); // most likely not this
	// cfg.reset(nullptr);
}

void FaceVacsEngine::enrollGallery(std::vector<facerecognition::FaceRecord> galleryRecords, path databasePath)
{
	// We first enroll the whole gallery:
	auto logger = Loggers->getLogger("facerecognition");
	logger.info("Enroling gallery, " + std::to_string(galleryRecords.size()) + " records...");

	ThreadPool threadPool(4);
	vector<std::future<boost::optional<path>>> firFutures;

	// create an enrollment processor
	FRsdk::Enrollment::Processor proc(*cfg);
	// Todo: FaceVACS should be able to do batch-enrolment?
	auto cnt = 0;
	for (auto& r : galleryRecords) {
		++cnt;
		logger.info("Enroling gallery image " + std::to_string(cnt));
		// Does the FIR already exist in the temp-dir? If yes, skip FD/EyeDet!
		auto firPath = tempDir / r.dataPath;
		firPath.replace_extension(".fir");
		if (boost::filesystem::exists(firPath)) {
			continue;
		}
		// Create the FIR:
		// We can't enqueue createFir directly because it has overloads. std::async/std::bind suffer
		// from the same. Options: Casting, lambda, a wrapper struct.
		firFutures.emplace_back(threadPool.enqueue([this](const path& s) { return createFir(s); }, databasePath / r.dataPath));
		//boost::optional<path> fir = createFir(databasePath / r.dataPath);
	}
	// Wait for all the threads.
	// We don't check for boost::none, we don't reuse the path anyway.
	// A boost::none means we couldn't create the .fir, i.e. couldn't enrol the image
	for (auto& f : firFutures) {
		//f.get();
		auto fir = f.get();
		//if (!fir) {
		//	continue;
		//}
	}

	// Enrol all the FIRs now:
	for (auto& r : galleryRecords) {
		auto firPath = tempDir / r.dataPath;
		firPath.replace_extension(".fir");
		std::ifstream firIn(firPath.string(), std::ios::in | std::ios::binary);
		// Some of the FIRs might not exist due to enrolment failure
		if (firIn.is_open() && firIn.good()) {
			try {
				auto fir = firBuilder->build(firIn);
				population->append(fir, firPath.string());
				this->enrolledGalleryRecords.push_back(r); // we make a copy of the record? not sure?
			}
			// Catch the following here - we can't continue.
			catch (const FRsdk::FeatureDisabled& e) {
				logger.error(e.what());
				// Terminate/exit? No, well we shouldn't do that here. We should abort in main, so all the d'tors get
				// called. I think we should re-throw a generic, own fr::enrollment_exception here that every Engine class shares.
				// That also gives a chance for the calling code to handle it!
			}
			catch (const FRsdk::LicenseSignatureMismatch& e) {
				logger.error(e.what());
			}
			catch (std::exception& e) { // I think we shouldn't catch this here
				logger.error(e.what());
			}
		}
	}
	//return enrolledGalleryRecords;
}

float FaceVacsEngine::match(path g, std::string gallerySubjectId, cv::Mat p, std::string probeFilename, cv::Vec2f firstEye, cv::Vec2f secondEye)
{
	auto logger = Loggers->getLogger("facerecognition");

	auto galleryFirPath = tempDir / g.filename();
	galleryFirPath.replace_extension(".fir"); // we probably don't need this - only the basename / subject ID (if it is already enroled)

	// TODO: 2 versions of this function: path and Mat. Change the Mat version => remove imwrite.
	auto tempProbeImageFilename = tempDir / path(probeFilename).filename();
	cv::imwrite(tempProbeImageFilename.string(), p);

	boost::optional<path> probeFrameFir = createFir(tempProbeImageFilename);
	if (!probeFrameFir) {
		logger.info("Couldn't enroll the given probe - not a good frame. Returning score 0.0.");
		return 0.0;
	}

	std::ifstream firStream(probeFrameFir->string(), std::ios::in | std::ios::binary);
	FRsdk::FIR fir = firBuilder->build(firStream);

	//compare() does not care about the configured number of threads
	//for the comparison algorithm. It uses always one thread to
	//compare all in order to preserve the order of the scores
	//according to the order in the population (order of adding FIRs to
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
}

std::vector<std::pair<FaceRecord, float>> FaceVacsEngine::matchAll(cv::Mat p, std::string probeFilename, cv::Vec2f firstEye, cv::Vec2f secondEye)
{
	auto tempProbeImageFilename = tempDir / path(probeFilename).filename();
	tempProbeImageFilename.replace_extension(".fir");
	boost::optional<path> probeFrameFir = createFir(matToFRsdkImage(p), tempProbeImageFilename);
	if (!probeFrameFir) {
		std::cout << "Couldn't enroll the probe - not a good frame. Return an empty vector." << std::endl;
		return std::vector < std::pair<FaceRecord, float> > {};
	}

	std::ifstream firStream(probeFrameFir->string(), std::ios::in | std::ios::binary);
	FRsdk::FIR fir = firBuilder->build(firStream);

	//compare() does not care about the configured number of Threads
	//for the comparison algorithm. It uses always one thread to
	//compare all in order to preserve the order of the scores
	//according to the order in the population (order of adding FIRs to
	//the population)
	FRsdk::CountedPtr<FRsdk::Scores> scores = me->compare(fir, *population);
	//std::vector<float>{ std::make_move_iterator(std::begin(*scores)), std::make_move_iterator(std::end(*scores)) };

	std::vector<std::pair<FaceRecord, float>> ret;
	ret.reserve(enrolledGalleryRecords.size());
	size_t i = 0;
	for (auto& e : *scores) {
		ret.emplace_back(std::make_pair(enrolledGalleryRecords[i], e));
		++i;
	}
	return ret;
}

std::vector<std::pair<FaceRecord, float>> FaceVacsEngine::matchAll(path probeFilename, cv::Vec2f firstEye, cv::Vec2f secondEye)
{
	//auto tempProbeImageFilename = tempDir / path(probeFilename).filename(); // img already on fs, no need to create temporary in ./fvsdk_tmp. Pass orig FN.
	//tempProbeImageFilename.replace_extension(".fir"); // difference to other matchAll function. Pass img-fn, not fir-fn.
	boost::optional<path> probeFrameFir = createFir(probeFilename); // difference to other matchAll function
	if (!probeFrameFir) {
		std::cout << "Couldn't enroll the probe - not a good frame. Return an empty vector." << std::endl;
		return std::vector<std::pair<FaceRecord, float>>{};
	}

	std::ifstream firStream(probeFrameFir->string(), std::ios::in | std::ios::binary);
	FRsdk::FIR fir = firBuilder->build(firStream);

	//compare() does not care about the configured number of Threads
	//for the comparison algorithm. It uses always one thread to
	//compare all in order to preserve the order of the scores
	//according to the order in the population (order of adding FIRs to
	//the population)
	FRsdk::CountedPtr<FRsdk::Scores> scores = me->compare(fir, *population);
	//std::vector<float>{ std::make_move_iterator(std::begin(*scores)), std::make_move_iterator(std::end(*scores)) };

	std::vector<std::pair<FaceRecord, float>> ret;
	ret.reserve(enrolledGalleryRecords.size());
	size_t i = 0;
	for (auto& e : *scores) {
		ret.emplace_back(std::make_pair(enrolledGalleryRecords[i], e));
		++i;
	}
	return ret;
}

boost::optional<path> FaceVacsEngine::createFir(path image)
{
	// Does the FIR already exist in the temp-dir? If yes, skip FD/EyeDet!
	auto firPath = tempDir / image.filename();
	firPath.replace_extension(".fir");
	if (boost::filesystem::exists(firPath)) {
		//throw std::runtime_error("Unexpected. Should check for that.");
		// just overwrite
		return firPath;
	}

	return createFir(FRsdk::ImageIO::load(image.string()), firPath);
}

boost::optional<path> FaceVacsEngine::createFir(FRsdk::Image image, path firPath)
{
	auto logger = Loggers->getLogger("facerecognition");
	// create an enrollment processor
	FRsdk::Enrollment::Processor proc(*cfg);
	// Todo: FaceVACS should be able to do batch-enrolment?

	logger.info("Trying to create the FIR: " + firPath.string());

	FRsdk::SampleSet enrollmentImages;
	auto sample = FRsdk::Sample(image);
	// Once we have pasc-still-training landmarks from Ross, we can use those here. In the mean-time, use the Cog Eyefinder:
	FRsdk::Face::Finder faceFinder(*cfg);
	FRsdk::Eyes::Finder eyesFinder(*cfg);
	float mindist = 0.005f; // minEyeDist, def = 0.1; me: 0.01; philipp: 0.005
	float maxdist = 0.3f; // maxEyeDist, def = 0.4; me: 0.3; philipp: 0.4
	FRsdk::Face::LocationSet faceLocations = faceFinder.find(image, mindist, maxdist);
	if (faceLocations.empty()) {
		logger.info("FaceFinder: No face found.");
		return boost::none;
	}
	// We just use the first found face:
	// doing eyes finding
	FRsdk::Eyes::LocationSet eyesLocations = eyesFinder.find(image, *faceLocations.begin());
	if (eyesLocations.empty()) {
		logger.info("EyeFinder: No eyes found.");
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
}

} /* namespace facerecognition */
