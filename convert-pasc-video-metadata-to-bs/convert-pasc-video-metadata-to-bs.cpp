/*
 * convert-pasc-metadata.cpp
 *
 *  Created on: 11.09.2014
 *      Author: Patrik Huber
 *
 * Read in PaSC video detections and metadata information from a
 * pasc_video_pittpatt_detections.csv or pasc_training_video_pittpatt_detection.csv
 * file and serialise it to a text file using boost::serialize.
 * Example:
 * convert-pasc-metadata -i pasc_video_pittpatt_detections.csv -o out/
 *   
 */

#include <memory>
#include <iostream>
#include <fstream>

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem.hpp"
#include "boost/archive/text_oarchive.hpp"
//#include "boost/archive/text_iarchive.hpp"
//#include "boost/archive/binary_oarchive.hpp"
//#include "boost/archive/binary_iarchive.hpp"
#include "boost/serialization/optional.hpp"
#include "boost/serialization/vector.hpp"

#include "facerecognition/pasc.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using boost::filesystem::path;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::make_shared;

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path inputFile;
	path outputFolder;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("input,i", po::value<path>(&inputFile)->required(),
				"pasc_video_pittpatt_detections.csv or pasc_training_video_pittpatt_detection.csv file")
			("output,o", po::value<path>(&outputFolder)->default_value("."),
				"path to an output folder")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: convert-pasc-metadata [options]" << endl;
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
	
	Loggers->getLogger("facerecognition").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Create the output directory if it doesn't exist yet
	if (!boost::filesystem::exists(outputFolder)) {
		boost::filesystem::create_directory(outputFolder);
	}

	auto videoDetections = facerecognition::readPascVideoDetections(inputFile);
	auto outputFilename = outputFolder / inputFile.filename();
	outputFilename.replace_extension(".txt");
	std::ofstream ofPascT(outputFilename.string());
	{ // use scope to ensure archive goes out of scope before stream
		boost::archive::text_oarchive oa(ofPascT);
		oa << videoDetections;
	}
	ofPascT.close();
	
	// Same, but binary file instead. Note: The writing works but the reading gives a runtime crash.
/*	outputFilename.replace_extension(".bin");
	std::ofstream ofPascB(outputFilename.string());
	{
		boost::archive::binary_oarchive oa(ofPascB);
		oa << videoDetections;
		// archive and stream closed when destructors are called
	}
*/
	return EXIT_SUCCESS;
}
