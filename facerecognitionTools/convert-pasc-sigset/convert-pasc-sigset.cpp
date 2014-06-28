/*
 * convertPascSigset.cpp
 *
 *  Created on: 23.06.2014
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

#include "facerecognition/FaceRecord.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/property_tree/xml_parser.hpp"

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
			cout << "Usage: generateMultipieList [options]\n";
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
	Loggers->getLogger("convertPascSigset").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("convertPascSigset");
	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	path inputSigset(R"(C:\Users\Patrik\Documents\GitHub\experiments\PaSC\lists\still.xml)");
	path outputSigset(R"(C:\Users\Patrik\Documents\GitHub\experiments\PaSC\lists\still.sig.txt)");

	ptree input;
	try {
		boost::property_tree::read_xml(inputSigset.string(), input);
	}
	catch (boost::property_tree::xml_parser_error& e) {
		appLogger.error("..");
		return EXIT_FAILURE;
	}
	
	ptree sigset; // The output tree
	//sigset.put("database", "MultiPIE");
	
	ptree sigs = input.get_child("biometric-signature-set");
	for (auto&& e : sigs) {
		if (e.first != "<xmlattr>") {
			string bio_sig_name = e.second.get<string>("<xmlattr>.name");
			ptree bio_sig_presentation = e.second.get_child("presentation");
			string pres_name = bio_sig_presentation.get<string>("<xmlattr>.name");
			//string pres_modality = bio_sig_presentation.get<string>("<xmlattr>.modality");
			string pres_filename = bio_sig_presentation.get<string>("<xmlattr>.file-name");
			//string pres_fileformat = bio_sig_presentation.get<string>("<xmlattr>.file-format");

			facerecognition::FaceRecord faceRecord;
			faceRecord.identifier = pres_name;
			faceRecord.subjectId = bio_sig_name;
			faceRecord.dataPath = pres_filename;
			//faceRecord.other = pres_name; // Use this or the filename as identifier?
			ptree entry = facerecognition::FaceRecord::convertTo(faceRecord);
			sigset.add_child("records.record", entry);
		}
		
	}
	
	boost::property_tree::write_info(outputSigset.string(), sigset);
	return 0;
}
