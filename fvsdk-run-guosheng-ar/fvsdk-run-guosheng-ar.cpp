/*
 * fvsdk-run-guosheng-ar.cpp
 *
 *  Created on: 08.11.2014
 *      Author: Patrik Huber
 *
 * Example:
 * fvsdk-run-guosheng-ar -s "Z:/datasets/multiview02/PaSC/Protocol/PaSC_20130611/PaSC/metadata/sigsets/pasc_video_handheld.xml" -f "Z:\FRonPaSC\patrik\video_handheld_fvsdk_allFrames_vs_allFrames\FIRs"
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

#include "frsdk/enroll.h"
#include "frsdk/match.h"

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

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path probeDirectory, galleryDirectory;
	path outputFile;
	path fvsdkConfig;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("probe-dir,p", po::value<path>(&probeDirectory)->required(),
				  "PaSC video sigset. Used as query and target.")
			("gallery-dir,g", po::value<path>(&galleryDirectory)->required(),
				"path to the pre-enroled FIRs")
			("output,o", po::value<path>(&outputFile)->default_value("./similarity_matrix.csv"),
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
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	if (!boost::filesystem::exists(galleryDirectory)) {
		appLogger.error("The given input FIRs directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}

	vector<path> probeImagePaths, galleryImagePaths;
	std::copy(fs::directory_iterator(probeDirectory), fs::directory_iterator(), std::back_inserter(probeImagePaths));
	std::copy(fs::directory_iterator(galleryDirectory), fs::directory_iterator(), std::back_inserter(galleryImagePaths));

	// Same order as in filesystem:
	std::sort(begin(probeImagePaths), end(probeImagePaths));
	std::sort(begin(galleryImagePaths), end(galleryImagePaths));

	vector<Mat> probeImages;// , galleryImages;
	vector<facerecognition::FaceRecord> galleryImages;
	for (auto& p : probeImagePaths) {
		//probeImages.emplace_back(cv::imread(p.string()));
	}
	for (auto& g : galleryImagePaths) {
		//galleryImages.emplace_back(cv::imread(g.string()));
		auto r = facerecognition::FaceRecord();
		r.identifier = (g.parent_path().leaf() / g.filename()).string();
		r.subjectId = g.filename().string();
		r.dataPath = g.parent_path().leaf() / g.filename();
		galleryImages.emplace_back(r);
	}

	facerecognition::FaceVacsEngine faceRecEngine(fvsdkConfig, "./tmp_fvsdk"); // temp-dir not needed, we only match
	auto engineCfg = faceRecEngine.getConfigurationInstance();
	auto firBuilder = std::unique_ptr<FRsdk::FIRBuilder>(new FRsdk::FIRBuilder(*engineCfg.get()));
	auto matchingEngine = std::unique_ptr<FRsdk::FacialMatchingEngine>(new FRsdk::FacialMatchingEngine(*engineCfg.get()));

	Mat fullSimilarityMatrix = Mat::zeros(probeImagePaths.size(), galleryImagePaths.size(), CV_32FC1);
	
	path dataRoot = galleryDirectory.parent_path();
	faceRecEngine.enrollGallery(galleryImages, dataRoot);
	// auto population = faceRecEngine.getPopulation(); // for rank 1 id rate
	int numRank1IdMatches = 0;
	for (size_t p = 0; p < probeImagePaths.size(); ++p)
	{
		appLogger.info("Matching probe " + std::to_string(p + 1) + "...");
		// Simi Mtx:
		auto scores = faceRecEngine.matchAll(probeImagePaths[p], cv::Vec2f(), cv::Vec2f());
		if (scores.size() != 100) {
			throw std::runtime_error("Couldn't enrol probe, fix me!");
		}
		int gallery = 0;
		for (auto s : scores) {
			fullSimilarityMatrix.at<float>(p, gallery) = static_cast<float>(s.second);
			++gallery;
		}
		// Rank-1 id rate:
	/*	path probeFrameFir = path("./tmp_fvsdk") / probeImagePaths[p].parent_path().leaf() / probeImagePaths[p].filename();
		probeFrameFir.replace_extension(".fir");
		std::ifstream firStream(probeFrameFir.string(), std::ios::in | std::ios::binary);
		FRsdk::FIR fir = firBuilder->build(firStream);
		auto result = matchingEngine->bestMatches(fir, *population, FRsdk::Score(0.0f), 1);
		auto a = probeImagePaths[p].stem().string();
		auto b = path((*result).front().first).stem().string(); // dereference, take first of list, first elem = firname, take basename of it (which will be e.g. 'M-001')
		if (a == b) {
			++numRank1IdMatches;
		}
	*/
	} // end for over all query videos

	appLogger.info("Rank 1 matches: " + std::to_string(numRank1IdMatches) + " out of 100 images.");

	//appLogger.info("Finished filling the full similarity matrix. Saving as CSV...");
	//facerecognition::utils::saveSimilarityMatrixAsCSV(fullSimilarityMatrix, outputFile);
	//appLogger.info("Successfully saved PaSC CSV similarity matrix.");

	return EXIT_SUCCESS;
}
