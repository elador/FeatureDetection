/*
 * fvsdk-match-video-to-video-allframes.cpp
 *
 *  Created on: 11.10.2014
 *      Author: Patrik Huber
 *
 * Example:
 * match-all-frames -s "Z:/datasets/multiview02/PaSC/Protocol/PaSC_20130611/PaSC/metadata/sigsets/pasc_video_handheld.xml" -f "Z:\FRonPaSC\patrik\video_handheld_fvsdk_allFrames_vs_allFrames\FIRs"
 *   
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>

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

#include "frsdk/enroll.h"
#include "frsdk/match.h"

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
	path firDirectory;
	path sigsetFile;
	path outputFile;
	path fvsdkConfig;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("sigset,s", po::value<path>(&sigsetFile)->required(),
				  "PaSC video sigset. Used as query and target.")
			("firs,f", po::value<path>(&firDirectory)->required(),
				"path to the pre-enroled FIRs")
			("output,o", po::value<path>(&outputFile)->default_value("./output.csv"),
				"output similarity matrix")
			("fvsdk-config,c", po::value<path>(&fvsdkConfig)->default_value(R"(C:\FVSDK_8_9_5\etc\frsdk.cfg)"),
				"path to frsdk.cfg. Usually something like C:\\FVSDK_8_9_5\\etc\\frsdk.cfg")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: match-all-frames [options]" << std::endl;
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
	Loggers->getLogger("match-all-frames").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("match-all-frames");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	if (!boost::filesystem::exists(firDirectory)) {
		appLogger.error("The given input FIRs directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}

	// Read the training-video xml sigset and the training-still sigset to get the subject-id metadata:
	auto videoSigset = facerecognition::utils::readPascSigset(sigsetFile, true);

	facerecognition::FaceVacsEngine faceRecEngine(fvsdkConfig, "./tmp_fvsdk"); // temp-dir not needed, we only match
	auto engineCfg = faceRecEngine.getConfigurationInstance();
	auto firBuilder = std::unique_ptr<FRsdk::FIRBuilder>(new FRsdk::FIRBuilder(*engineCfg.get()));
	auto matchingEngine = std::unique_ptr<FRsdk::FacialMatchingEngine>(new FRsdk::FacialMatchingEngine(*engineCfg.get()));

	Mat fullSimilarityMatrix = Mat::zeros(videoSigset.size(), videoSigset.size(), CV_32FC1);
	// Later on, targetFirFiles and queryFirFiles could be empty if no frame of a video could be
	// enroled. We'll just skip over them and leave these entries 0.

	// Preload all FIRs, i.e. only ever read all of them once, for speed (profiling showed bottleneck):
	// We can only pre-select 14. 14*1401 <= 20'000 max allowed FIR instances.
	//std::map<string, FRsdk::FIR> firs;
	std::map<string, vector<std::pair<string, FRsdk::FIR>>> firs;
	for (size_t q = 0; q < videoSigset.size(); ++q)
	{
		//if (q > 2) {
		//	break;
		//}
		appLogger.info("Loading FIRs of video " + std::to_string(q + 1) + " of " + std::to_string(videoSigset.size()) + "...");
		auto queryVideoFirDirectory = firDirectory / videoSigset[q].dataPath.stem();
		if (!boost::filesystem::exists(queryVideoFirDirectory)) {
			appLogger.info("No FIRs for this query video. No frames could be enrolled.");
			throw std::runtime_error("Does this happen? I think the enrol app always creates the folder?");
		}
		vector<path> queryFirFiles;
		std::copy(fs::directory_iterator(queryVideoFirDirectory), fs::directory_iterator(), std::back_inserter(queryFirFiles));

		int numFirsPerVideo = 2; // max 14 for <= 20'000 FIRs
		vector<size_t> randomIndices(queryFirFiles.size());
		std::iota(begin(randomIndices), end(randomIndices), 0);
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(begin(randomIndices), end(randomIndices), g);
		randomIndices.resize(numFirsPerVideo);
		
		vector<std::pair<string, FRsdk::FIR>> firsThisVideo;
		//for (auto& firPath : queryFirFiles) {
		for (size_t i = 0; i < randomIndices.size(); ++i) {
			std::ifstream firIn(queryFirFiles[randomIndices[i]].string(), std::ios::in | std::ios::binary);
			if (firIn.is_open() && firIn.good()) {
				firsThisVideo.push_back(std::make_pair(queryFirFiles[randomIndices[i]].stem().string(), firBuilder->build(firIn)));
				//population->append(fir, firPath.stem().string());
				// extract & push back frame num:
				//populationIdentifiers.emplace_back(firPath.stem().string());
			}
		}

		firs.emplace(videoSigset[q].dataPath.stem().string(), firsThisVideo);
	}

	for (size_t q = 0; q < videoSigset.size(); ++q)
	{
		appLogger.info("Matching query video " + std::to_string(q + 1) + " of " + std::to_string(videoSigset.size()) + "...");
		auto queryVideoFirDirectory = firDirectory / videoSigset[q].dataPath.stem();
		if (!boost::filesystem::exists(queryVideoFirDirectory)) {
			appLogger.info("No FIRs for this query video. No frames could be enrolled. Setting every score to 0.0.");
			throw std::runtime_error("Does this happen? I think the enrol app always creates the folder?");
		}
		//vector<path> queryFirFiles;
		//std::copy(fs::directory_iterator(queryVideoFirDirectory), fs::directory_iterator(), std::back_inserter(queryFirFiles));

		for (size_t t = q; t < videoSigset.size(); ++t) // start at t = q to only match the upper diagonal (scores are symmetric)
		{
			appLogger.info("... against target video " + std::to_string(t + 1) + " of " + std::to_string(videoSigset.size()));
			auto targetVideoFirDirectory = firDirectory / videoSigset[t].dataPath.stem();
			if (!boost::filesystem::exists(targetVideoFirDirectory)) {
				appLogger.info("No FIRs for this target video. No frames could be enrolled. Setting every score to 0.0.");
				throw std::runtime_error("Does this happen? I think the enrol app always creates the folder?");
			}

			// Read in all FIRs, with frame-number, put in gallery:
			// We extract the frame-number too, but only for informational purposes (e.g. to be able to visually inspect it)
			//vector<path> targetFirFiles;
			//std::copy(fs::directory_iterator(targetVideoFirDirectory), fs::directory_iterator(), std::back_inserter(targetFirFiles));

			// Match every query frame against every target frame.
			// We do this by enroling all target frames as gallery:
			auto population = std::unique_ptr<FRsdk::Population>(new FRsdk::Population(*engineCfg.get()));
			vector<string> populationIdentifiers;
			auto targetFirsThisVideo = firs.find(videoSigset[t].dataPath.stem().string()); // pre-loaded
			//for (auto& firPath : targetFirFiles) {
			for (auto& targetFir : targetFirsThisVideo->second) {
				//std::ifstream firIn(targetFir.first, std::ios::in | std::ios::binary);
				//if (firIn.is_open() && firIn.good()) {
					//auto fir = firBuilder->build(firIn);
					//population->append(fir, path(targetFir.first).stem().string());
					population->append(targetFir.second, targetFir.first);
					// extract & push back frame num:
					populationIdentifiers.emplace_back(targetFir.first);
				//}
			}
			// For every query frame:
			typedef std::tuple<string, string, float> MatchingPair; // queryFrame, targetFrame, score
			vector<MatchingPair> allScores;
			auto queryFirsThisVideo = firs.find(videoSigset[q].dataPath.stem().string()); // pre-loaded
			//for (auto& queryFrameFirPath : queryFirFiles) {
			for (auto& queryFir : queryFirsThisVideo->second) {
				// Match against gallery (=all target frames)
				//std::ifstream firStream(queryFrameFirPath.string(), std::ios::in | std::ios::binary);
				//FRsdk::FIR queryFrameFir = firBuilder->build(firStream);
				//FRsdk::CountedPtr<FRsdk::Scores> scores = matchingEngine->compare(queryFrameFir, *population);
				FRsdk::CountedPtr<FRsdk::Scores> scores = matchingEngine->compare(queryFir.second, *population);
				// Add all scores for this query frame as tuple to the allScores vector:
				int ele = 0;
				for (auto s : *scores) {
					allScores.emplace_back(std::make_tuple(queryFir.first, populationIdentifiers[ele], s));
					++ele;
				}
			}
			// We now got all scores of all query frames of one video against all target frames of a target video.
			// Do some selection magic (or save result?)
			// Note/Todo: allScores could be empty?
			if (allScores.size() != 0) {
				auto maxScoreIter = std::max_element(begin(allScores), end(allScores), [](const MatchingPair& a, const MatchingPair& b) {return (std::get<2>(a) < std::get<2>(b)); });
				//auto maxScoreIndex = std::distance(begin(allScores), maxScoreIter);
				fullSimilarityMatrix.at<float>(q, t) = std::get<2>(*maxScoreIter);
				fullSimilarityMatrix.at<float>(t, q) = std::get<2>(*maxScoreIter); // fill the lower diagonal - scores are symmetric
			}
			// otherwise, leave entries at 0 (initialised value)
		} // end for over all target videos
	} // end for over all query videos

	appLogger.info("Finished filling the full similarity matrix. Saving as CSV...");
	facerecognition::utils::saveSimilarityMatrixAsCSV(fullSimilarityMatrix, outputFile);
	appLogger.info("Successfully saved PaSC CSV similarity matrix.");

	return EXIT_SUCCESS;
}
