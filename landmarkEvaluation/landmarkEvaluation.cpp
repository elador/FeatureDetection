/*
* landmarkEvaluation.cpp
*
*  Created on: 03.02.2014
*      Author: Patrik Huber
*/

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#ifdef WIN32
#include <SDKDDKVer.h>
#endif

/*	// There's a bug in boost/optional.hpp that prevents us from using the debug-crt with it
// in debug mode in windows. It works in release mode, but as we need debugging, let's
// disable the windows-memory debugging for now.
#ifdef WIN32
#include <crtdbg.h>
#endif

#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG
*/

#include <chrono>
#include <memory>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/algorithm/string.hpp"

#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/IbugLandmarkFormatParser.hpp"
#include "imageio/SimpleModelLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using boost::property_tree::ptree;
using boost::filesystem::path;
using cv::Mat;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;


template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(cout, " "));
	return os;
}

int main(int argc, char *argv[])
{
#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
#endif

	string verboseLevelConsole;
	path configFilename, inputPath, groundtruthPath, outputFilename;
	string inputType, groundtruthType;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO", "show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("config,c", po::value<path>(&configFilename)->required(),
				"path to a config (.cfg) file")
			("input,i", po::value<path>(&inputPath)->required(),
				"input landmarks")
			("input-type,s", po::value<string>(&inputType)->required(),
				"specify the type of landmarks to load: simple")
			("groundtruth,g", po::value<path>(&groundtruthPath)->required(),
				"groundtruth landmarks")
			("groundtruth-type,t", po::value<string>(&groundtruthType)->required(),
				"specify the type of landmarks to load: ibug")
			("output,o", po::value<path>(&outputFilename),
				"output filename to write the normalized landmark errors to")
			;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: landmarkEvaluation [options]\n";
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
	if (boost::iequals(verboseLevelConsole, "PANIC")) logLevel = LogLevel::Panic;
	else if (boost::iequals(verboseLevelConsole, "ERROR")) logLevel = LogLevel::Error;
	else if (boost::iequals(verboseLevelConsole, "WARN")) logLevel = LogLevel::Warn;
	else if (boost::iequals(verboseLevelConsole, "INFO")) logLevel = LogLevel::Info;
	else if (boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = LogLevel::Debug;
	else if (boost::iequals(verboseLevelConsole, "TRACE")) logLevel = LogLevel::Trace;
	else {
		cout << "Error: Invalid loglevel." << endl;
		return EXIT_FAILURE;
	}

	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("landmarkEvaluation").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("landmarkEvaluation");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));
	appLogger.debug("Using config: " + configFilename.string());

	// output: the avg-err plus a file with the differences (+comments?) for matlab.

	// Load the ground truth
	shared_ptr<NamedLandmarkSource> groundtruthSource;
	vector<path> groundtruthDirs{ groundtruthPath };
	shared_ptr<LandmarkFormatParser> landmarkFormatParser;
	if (boost::iequals(groundtruthType, "ibug")) {
		landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
		groundtruthSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(nullptr, ".pts", GatherMethod::SEPARATE_FOLDERS, groundtruthDirs), landmarkFormatParser);
	}
	else {
		appLogger.error("Error: Invalid ground-truth landmarks type.");
		return EXIT_FAILURE;
	}

	// Load the landmarks to compare against, usually automatically detected landmarks:
	shared_ptr<NamedLandmarkSource> inputSource;
	vector<path> inputDirs{ inputPath };
	shared_ptr<LandmarkFormatParser> landmarkFormatParserInput;
	if (boost::iequals(inputType, "ibug")) {
		landmarkFormatParserInput = make_shared<IbugLandmarkFormatParser>();
		inputSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(nullptr, ".pts", GatherMethod::SEPARATE_FOLDERS, inputDirs), landmarkFormatParserInput);
	}
	else if (boost::iequals(inputType, "simple")) {
		landmarkFormatParserInput = make_shared<SimpleModelLandmarkFormatParser>();
		inputSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(nullptr, ".txt", GatherMethod::SEPARATE_FOLDERS, inputDirs), landmarkFormatParserInput);
	}
	else {
		appLogger.error("Error: Invalid input landmarks type.");
		return EXIT_FAILURE;
	}

	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(configFilename.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	// Create the output file if one was specified on the command line
	std::ofstream outputFile;
	if (!outputFilename.empty()) {
		outputFile.open(outputFilename.string());
		if (!outputFile.is_open()) {
			appLogger.error("An output filename was specified, but the file could not be created: " + outputFilename.string());
			return EXIT_FAILURE;
		}
	}

	vector<float> differences;

	appLogger.info("Starting to process...");

	// Loop over the input landmarks
	while (inputSource->next())	{
		path filename = inputSource->getName();
		LandmarkCollection detected = inputSource->getLandmarks();
		LandmarkCollection groundtruth;
		try {
			groundtruth = groundtruthSource->get(filename);
		}
		catch (std::out_of_range& e) {
			appLogger.warn("Skipping this input file. This is probably not expected.");
			continue;
		}
		

		float difference = 0.3f;
		differences.push_back(difference);
		if (!outputFilename.empty()) {
			outputFile << difference << endl;
		}

	}
	
	// close the file if we had opened it before
	if (!outputFilename.empty()) {
		outputFile.close();
	}

	appLogger.info("Finished processing all input landmarks.");

	return 0;
}
