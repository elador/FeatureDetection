/*
 * train-deeplearning-matcher-extract.cpp
 *
 *  Created on: 01.11.2014
 *      Author: Patrik Huber
 *
 * Goal: Preprocessing.
 *
 * Example:
 * train-deeplearning-matcher ...
 *   
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>

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
#include "boost/archive/text_oarchive.hpp"
#include "boost/serialization/vector.hpp"

#include "imageio/MatSerialization.hpp"

#include "facerecognition/utils.hpp"

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
	path sigsetFile;
	path inputPatchesDirectory;
	path outputFile;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("sigset,s", po::value<path>(&sigsetFile)->required(),
				  "PaSC video sigset")
			("data-path,d", po::value<path>(&inputPatchesDirectory)->required(),
				"path to video frames, from sigset, cropped")
			("output,o", po::value<path>(&outputFile)->default_value("."),
				"output file for the data, in boost::serialization text format. Folder needs to exist.")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: train-deeplearning-matcher-extract [options]" << std::endl;
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
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Create the output directory if it doesn't exist yet:
	//if (!boost::filesystem::exists(outputFile)) {
	//	boost::filesystem::create_directory(outputFile);
	//}
	
	// The training data:
	vector<Mat> trainingData; // Will have the same size and order of the sigset

	auto sigset = facerecognition::utils::readPascSigset(sigsetFile, true);

	std::map<path, path> filesMap;
	{
		vector<path> files;
		try	{
			std::copy(fs::directory_iterator(inputPatchesDirectory), fs::directory_iterator(), std::back_inserter(files));
		}
		catch (const fs::filesystem_error& ex)
		{
			appLogger.error(ex.what());
			return EXIT_FAILURE;
		}
		// Convert to a map - otherwise, the search will be be very slow.
		for (auto& f : files) {
			filesMap.emplace(std::make_pair(f.stem().stem(), f));
		}
	}
	
	for (auto& s : sigset) {
		auto queryIter = filesMap.find(s.dataPath.stem());
		if (queryIter == end(filesMap)) {
			// no frame for this sigset entry...
			trainingData.emplace_back(Mat());
			continue;
		}
		// Todo/Note: What if we have several frames? Will the iterator just point to the first?
		Mat patch = cv::imread((inputPatchesDirectory / queryIter->second.filename()).string());
		// Here, we'd convert to gray, extract features etc.
		// Also we might read from the uncropped images, read in landmarks as well, and extract precise local features
		//cout << static_cast<float>(patch.rows) / patch.cols << endl;
		float aspect = 1.09f;
		cvtColor(patch, patch, cv::COLOR_RGB2GRAY);
		// (30, 33) would be better but because of tiny-cnn we choose 32x32 atm
		// No, we only have half the space, i.e. 1024 / 2 = 512 for one patch
		cv::resize(patch, patch, cv::Size(22, 23)); // could choose interpolation method
		patch = patch.reshape(1, 1); // Reshape to 1 row (row-vector - better suits OpenCV memory layout?)
		cv::hconcat(patch, cv::Mat::zeros(1, 6, CV_8UC1), patch); // fill rest from (22 * 23) to 512 with zeros
		trainingData.emplace_back(patch);
	}
	
	std::ofstream ofFrames(outputFile.string());
	{ // use scope to ensure archive goes out of scope before stream
		boost::archive::text_oarchive oa(ofFrames);
		oa << trainingData;
	}
	ofFrames.close();

	return EXIT_SUCCESS;
}
