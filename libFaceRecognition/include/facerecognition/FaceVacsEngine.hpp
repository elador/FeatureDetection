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

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include "boost/optional.hpp"

#include "opencv2/core/core.hpp"

#include <iostream>
#include <memory>
#include <fstream>

// Forward declarations:
namespace FRsdk {
	class Position;
	class Image;
	class Configuration;
	class FIRBuilder;
	class FacialMatchingEngine;
	class Population;
	class FIR;
	namespace Eyes {
		struct Location;
	}
}

namespace facerecognition {

// for the CoutEnrol
//std::ostream& operator<<(std::ostream& o, const FRsdk::Position& p);

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
	FaceVacsEngine(boost::filesystem::path frsdkConfig, boost::filesystem::path tempDirectory);

	~FaceVacsEngine();

	// because it often stays the same
	// Returns a new vector of FaceRecord with all the ones that could be enroled
	void enrollGallery(std::vector<facerecognition::FaceRecord> galleryRecords, boost::filesystem::path databasePath);

	// Make a matchAgainstGallery or matchSingle function as well
	// matches a pair of images, Cog LMs
	double match(cv::Mat first, cv::Mat second)
	{
		return 0.0;
	};
	// Always pass in images/image-filenames. The "outside" should never have to worry
	// about FIRs or other intermediate representations.
	// Eyes: Currently unused, use PaSC-tr eyes in the future
	float match(boost::filesystem::path g, std::string gallerySubjectId, cv::Mat p, std::string probeFilename, cv::Vec2f firstEye, cv::Vec2f secondEye);

	// Return will have the same order as enrolled gallery subjects!
	// I think returning a vec<pair<FaceRecord, score>> would be better - done
	std::vector<std::pair<FaceRecord, float>> matchAll(cv::Mat p, std::string probeFilename, cv::Vec2f firstEye, cv::Vec2f secondEye);

	// Same as above, but the image is on the filesystem (e.g. from a frameselection)
	std::vector<std::pair<FaceRecord, float>> matchAll(boost::filesystem::path probeFilename, cv::Vec2f firstEye, cv::Vec2f secondEye);

	// matches a pair of images, given LMs as Cog init
	// match(fn, fn, ...)
	// match(Mat, Mat, ...)

	// creates the FIR in temp-dir. Returns path to FIR if successful.
	// Todo: Directly return FIR?
	// If the method doesn't depend on the class instance/state, make it static?
	boost::optional<boost::filesystem::path> createFir(boost::filesystem::path image);

	// Todos etc see one function above!
	// Todo: We can remove 'firPath' here once we replace EnrolCoutFeedback by a direct-FIR-return class
	boost::optional<boost::filesystem::path> createFir(FRsdk::Image image, boost::filesystem::path firPath);

private:
	boost::filesystem::path tempDir;
	std::unique_ptr<FRsdk::Configuration> cfg; // static? (recommendation by fvsdk doc)
	std::unique_ptr<FRsdk::FIRBuilder> firBuilder;
	std::unique_ptr<FRsdk::FacialMatchingEngine> me;
	std::unique_ptr<FRsdk::Population> population; // enrolled gallery FIRs, same order
public:
	std::vector<facerecognition::FaceRecord> enrolledGalleryRecords; // We keep this for the subject IDs

};

} /* namespace facerecognition */
#endif /* FACEVACSENGINE_HPP_ */
