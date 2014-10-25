/*
 * match-videos-single-frames.cpp
 *
 *  Created on: 24.10.2014
 *      Author: Patrik Huber
 *
 * Example:
 * match-videos-single-frames ...
 *   
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <random>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/text_oarchive.hpp"
//#include "boost/archive/binary_iarchive.hpp"
//#include "boost/archive/binary_oarchive.hpp"

#include "boost/serialization/vector.hpp"
#include "boost/serialization/optional.hpp"
#include "boost/serialization/utility.hpp"

#include "imageio/MatSerialization.hpp"
#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"
#include "facerecognition/FaceVacsEngine.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using cv::Mat;
using boost::filesystem::path;
using boost::lexical_cast;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::vector;
using std::string;

//#include <boost/config.hpp>
//#include <boost/filesystem/path.hpp>
//#include <boost/serialization/level.hpp>
#include "facerecognition/PathSerialization.hpp"

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path inputDirectoryQueryVideos, inputDirectoryTargetVideos;
	path querySigset, targetSigset;
	path queryLandmarks;
	path outputPath;
	path fvsdkConfig;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("query-sigset,q", po::value<path>(&querySigset)->required(),
				  "PaSC video training query sigset")
			("query-path,r", po::value<path>(&inputDirectoryQueryVideos)->required(),
				"path to the training videos")
			("query-landmarks,l", po::value<path>(&queryLandmarks)->required(),
				"landmarks for the training videos in boost::serialization text format")
			("target-sigset,t", po::value<path>(&targetSigset)->required(),
				"PaSC still training target sigset")
			("target-path,u", po::value<path>(&inputDirectoryTargetVideos)->required(),
				"path to the training still images")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
			("fvsdk-config,c", po::value<path>(&fvsdkConfig)->default_value(R"(C:\FVSDK_8_9_5\etc\frsdk.cfg)"),
				"path to frsdk.cfg. Usually something like C:\\FVSDK_8_9_5\\etc\\frsdk.cfg")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: match-video [options]" << std::endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);

	}
	catch (po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	LogLevel logLevel;
	if(boost::iequals(verboseLevelConsole, "PANIC")) logLevel = LogLevel::Panic;
	else if(boost::iequals(verboseLevelConsole, "ERROR")) logLevel = LogLevel::Error;
	else if(boost::iequals(verboseLevelConsole, "WARN")) logLevel = LogLevel::Warn;
	else if(boost::iequals(verboseLevelConsole, "INFO")) logLevel = LogLevel::Info;
	else if(boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = LogLevel::Debug;
	else if(boost::iequals(verboseLevelConsole, "TRACE")) logLevel = LogLevel::Trace;
	else {
		cout << "Error: Invalid LogLevel." << endl;
		return EXIT_FAILURE;
	}
	
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("facerecognition").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("match-video").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("match-video");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	if (!boost::filesystem::exists(inputDirectoryQueryVideos)) {
		appLogger.error("The given input query video directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}
	if (!boost::filesystem::exists(inputDirectoryTargetVideos)) {
		appLogger.error("The given input target video directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}

	// Read the video detections metadata (eyes, face-coords):
	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(queryLandmarks.string());
		boost::archive::text_iarchive ia(ifs); // binary_iarchive
		ia >> pascVideoDetections;
	} // archive and stream closed when destructors are called

	// We would read the still images detection metadata (eyes, face-coords) here, but it's not provided with PaSC.
	// Todo: Try with and without the 5 Cog LMs

	// Read the training-video xml sigset and the training-still sigset to get the subject-id metadata:
	auto videoQuerySet = facerecognition::utils::readPascSigset(querySigset, true);
	auto videoTargetSet = facerecognition::utils::readPascSigset(targetSigset, true);

	// A little bit of a hack (until we change FaceRecord):
	// Replace the mp4 datapath with the image from the frame selection:
	if (inputDirectoryQueryVideos != inputDirectoryTargetVideos) {
		appLogger.error("Sorry, for now, query and target dir/sigsets must be the same. This would be a small change though, if needed.");
		return EXIT_FAILURE;
	}
	vector<path> files;
	try	{
		std::copy(fs::directory_iterator(inputDirectoryQueryVideos), fs::directory_iterator(), std::back_inserter(files));
	}
	catch (const fs::filesystem_error& ex)
	{
		appLogger.error(ex.what());
		return EXIT_FAILURE;
	}
	// Convert to a map - otherwise, the search will be be very slow.
	std::map<path, path> filesMap;
	for (auto& f : files) {
		filesMap.emplace(std::make_pair(f.stem().stem(), f));
	}
	
	for (auto& r : videoQuerySet) {
		auto queryIter = filesMap.find(r.dataPath.stem());
		if (queryIter == end(filesMap)) {
			// File not found on filesystem, means we don't have a frame for it. It will be a FTE.
			r.dataPath = path();
			continue;
		}
		r.dataPath = queryIter->second.filename();
	}

	// Create the output directory if it doesn't exist yet:
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	
	path fvsdkTempDir{ "./tmp_fvsdk" };
	if (!boost::filesystem::exists(fvsdkTempDir)) {
		boost::filesystem::create_directory(fvsdkTempDir);
	}
	facerecognition::FaceVacsEngine faceRecEngine(fvsdkConfig, fvsdkTempDir);
	videoQuerySet.resize(3); // 1000 = FIR limit atm
	faceRecEngine.enrollGallery(videoQuerySet, inputDirectoryTargetVideos); // NOTE: Should be videoTargetSet but they're identical

	// Loop over ALL the query sigset images (rows), then over each target (cols)
	Mat fullSimilarityMatrix(videoQuerySet.size(), videoQuerySet.size(), CV_32FC1);
	for (auto q = 0; q < videoQuerySet.size(); ++q) {
		appLogger.info("Matching row " + std::to_string(q));
		auto frameName = inputDirectoryQueryVideos / videoQuerySet[q].dataPath;
		// Scores from this query against all the (successfully enroled) target (gallery) frames
		auto recognitionScores = faceRecEngine.matchAll(frameName, cv::Vec2f(), cv::Vec2f());
		if (recognitionScores.size() == 0) {
			// we couldn't enrol the query frame
			fullSimilarityMatrix(cv::Range(q, q + 1), cv::Range(0, fullSimilarityMatrix.cols)) = cv::Scalar::all(0.0f); // rowRange, colRange. First is inclusive, second is exclusive.
			continue; // Skip the whole row, start with the next query image
		}
		for (auto t = 0; t < videoQuerySet.size(); ++t) { // ideally, target-sigset, not query
			// find videoQuerySet[t] in recognitionScores
			path targetImageNameFromSigset = videoQuerySet[t].dataPath;
			auto targetIter = std::find_if(begin(recognitionScores), end(recognitionScores), [targetImageNameFromSigset](const std::pair<facerecognition::FaceRecord, float>& p) { return (p.first.dataPath == targetImageNameFromSigset); });
			if (targetIter == end(recognitionScores)) {
				// File not found, means somewhere we had a FTE. We could set the whole column to 0.0f but we loop over it on the next row anyway so it doesn't matter if we do.
				fullSimilarityMatrix.at<float>(q, t) = 0.0f;
			}
			else {
				// Set to the score from recognitionScores
				fullSimilarityMatrix.at<float>(q, t) = targetIter->second;
			}
		}
	}

	appLogger.info("Finished filling the full similarity matrix. Saving as CSV...");
	facerecognition::utils::saveSimilarityMatrixAsCSV(fullSimilarityMatrix, outputPath / "similarityMatrix.csv");
	appLogger.info("Successfully saved PaSC CSV similarity matrix.");

	return EXIT_SUCCESS;
}
