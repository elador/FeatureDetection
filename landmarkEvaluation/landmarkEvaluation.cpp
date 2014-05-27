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
	path configFilename, inputPath, groundtruthPath;
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
			"specify the type of landmarks to load: ibug")
			("groundtruth,g", po::value<path>(&groundtruthPath)->required(),
			"gt landmarks")
			("groundtruth-type,t", po::value<string>(&groundtruthType)->required(),
			"specify the type of landmarks to load: ibug")
			;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << "Usage: landmarkEvaluation [options]\n";
			cout << desc;
			return EXIT_SUCCESS;
		}
	}
	catch (std::exception& e) {
		cout << e.what() << endl;
		return EXIT_FAILURE;
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

	Loggers->getLogger("landmarkEvaluation").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("landmarkEvaluation");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));
	appLogger.debug("Using config: " + configFilename.string());

	// TODO:
	// Gave up for now. Reasons:
	// - Not possible to loop over LandmarkSource
	// - No GatherMethod suitable. GatherMethod::SEPARATE_FILES maybe.

	// Load the ground truth
	shared_ptr<NamedLandmarkSource> groundtruthSource;
	vector<path> groundtruthDirs; groundtruthDirs.push_back(groundtruthPath);
	shared_ptr<LandmarkFormatParser> landmarkFormatParser;
	if (boost::iequals(groundtruthType, "lst")) {
		//landmarkFormatParser = make_shared<LstLandmarkFormatParser>();
		//landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, string(), GatherMethod::SEPARATE_FILES, groundtruthDirs), landmarkFormatParser);
	}
	else if (boost::iequals(groundtruthType, "ibug")) {
		landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
		groundtruthSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(nullptr, ".pts", GatherMethod::SEPARATE_FILES, groundtruthDirs), landmarkFormatParser);
	}
	else {
		cout << "Error: Invalid ground truth type." << endl;
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


	vector<imageio::ModelLandmark> landmarks;

	appLogger.info("Starting to process...");

	//while (groundtruthSource->get()) {
	
	//}

	return 0;
}
