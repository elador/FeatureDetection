/*
 * generateMatchlist.cpp
 *
 *  Created on: 16.06.2014
 *      Author: Patrik Huber
 */

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>

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

#include "logging/LoggerFactory.hpp"

#include "facerecognition/utils.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <exception>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using boost::filesystem::path;
using boost::property_tree::ptree;
using std::cout;
using std::endl;
using std::string;
using std::make_shared;
using std::vector;


int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(3759128);
	#endif
		
	string verboseLevelConsole;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO", "show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: generateMatchlist [options]\n";
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
		cout << "Error: Invalid LogLevel." << endl;
		return EXIT_FAILURE;
	}
	Loggers->getLogger("facerecognition").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("generateMatchlist").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("generateMatchlist");
	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	path probeSigsetFile{ R"(C:\Users\Patrik\Documents\GitHub\FeatureDetection\libFaceRecognition\share\sigset\MultiPIE_example.txt)" };
	path gallerySigsetFile{ R"(C:\Users\Patrik\Documents\GitHub\FeatureDetection\libFaceRecognition\share\sigset\MultiPIE_example.txt)" };
	path matchingConfigFile{ R"(C:\Users\Patrik\Documents\GitHub\FeatureDetection\libFaceRecognition\share\sigset\MultiPIE_example.txt)" };

	// Read the probe and gallery sigset files
	vector<facerecognition::FaceRecord> probeSigset = facerecognition::utils::readSigset(probeSigsetFile);
	vector<facerecognition::FaceRecord> gallerySigset = facerecognition::utils::readSigset(gallerySigsetFile);

	// Read the matching config file, and together with it, create the resulting matchlist:
	ptree matchingConfig;
	try {
		boost::property_tree::read_info(matchingConfigFile.string(), matchingConfig);
	}
	catch (boost::property_tree::info_parser_error& e) {
		string errorMessage{ string("Error reading the matching config file: ") + e.what() };
		appLogger.error(errorMessage);
		throw e;
	}

	// Create the matchlist
	ptree matchlist;

	for (auto&& probe : probeSigset) {
		ptree match;
		match.put("id", "001");
		match.put("img", "a.png");
		matchlist.add_child("images.image", match);
	}
				
	boost::property_tree::write_info("C:\\Users\\Patrik\\output_matchlist.txt", matchlist);
	
	return 0;
}
