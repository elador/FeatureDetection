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
using std::pair;

struct MatchFilePattern { // holds an entry of a matchlist (matching.txt) pattern, i.e. how to change the data file/paths to generate a Match
	string name; // original | basename: use the original name as given in the record or just the basename
	path rootPath;
	string prefix;
	string suffix;
	string replaceExtension;
};

path applyPattern(MatchFilePattern pattern, path dataPath)
{
	if (pattern.name == "original") {
		dataPath = dataPath;
	}
	else if (pattern.name == "basename") {
		dataPath = dataPath.filename();
	}
	string basename = dataPath.stem().string();
	string originalExtension = dataPath.extension().string();
	if (!pattern.prefix.empty()) {
		basename = pattern.prefix + basename;
	}
	if (!pattern.suffix.empty()) {
		basename = basename + pattern.suffix;
	}
	path filename(basename + originalExtension);
	if (!pattern.replaceExtension.empty()) {
		filename.replace_extension(pattern.replaceExtension);
	}
	path fullFilePath;
	if (pattern.name == "original") {
		auto lastSlash = dataPath.string().find_last_of("/\\");
		fullFilePath = pattern.rootPath / dataPath.string().substr(0, lastSlash) / filename;
	}
	else if (pattern.name == "basename") {
		fullFilePath = pattern.rootPath / filename;
	}
	return fullFilePath;
};

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

	path probeSigsetFile{ R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\probe_m30.sig.txt)" };
	path gallerySigsetFile{ R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\gallery.sig.txt)" };
	path matchingConfigFile{ R"(C:\Users\Patrik\Documents\GitHub\FeatureDetection\libFaceRecognition\share\config\matching.txt)" };
	path matchlistFile{ R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\matchlist_p_m30_g.txt)" }; // output

	// Read the probe and gallery sigset files
	vector<facerecognition::FaceRecord> probeSigset = facerecognition::utils::readSigset(probeSigsetFile);
	vector<facerecognition::FaceRecord> gallerySigset = facerecognition::utils::readSigset(gallerySigsetFile);

	// Read the matching config file, and together with it, create the resulting matchlist:
	ptree matchingConfig;
	try {
		boost::property_tree::read_info(matchingConfigFile.string(), matchingConfig);
	}
	catch (boost::property_tree::info_parser_error& error) {
		string errorMessage{ string("Error reading the matching config file: ") + error.what() };
		appLogger.error(errorMessage);
		throw error;
	}
	vector<MatchFilePattern> probeMatchPatterns;
	vector<MatchFilePattern> galleryMatchPatterns;
	try {
		ptree probeFiles = matchingConfig.get_child("matching.probe");
		for (auto&& f : probeFiles) {
			MatchFilePattern pattern;
			pattern.name = f.second.get<string>("name");
			pattern.rootPath = f.second.get<path>("rootPath");
			pattern.prefix = f.second.get<string>("prefix", "");
			pattern.suffix = f.second.get<string>("suffix", "");
			pattern.replaceExtension = f.second.get<string>("replaceExtension", "");
			probeMatchPatterns.emplace_back(pattern);
		}

		ptree galleryFiles = matchingConfig.get_child("matching.gallery");
		for (auto&& f : galleryFiles) {
			MatchFilePattern pattern;
			pattern.name = f.second.get<string>("name");
			pattern.rootPath = f.second.get<path>("rootPath");
			pattern.prefix = f.second.get<string>("prefix", "");
			pattern.suffix = f.second.get<string>("suffix", "");
			pattern.replaceExtension = f.second.get<string>("replaceExtension", "");
			galleryMatchPatterns.emplace_back(pattern);
		}
	}
	catch (const boost::property_tree::ptree_error& error) {
		string errorMessage{ string("Error parsing the matching config file: ") + error.what() };
		appLogger.error(errorMessage);
		throw error;
	}

	// Create the matchlist:
	// Generate the paths to every file specified in the matching config, according to the rules specified there
	vector<pair<string, vector<path>>> matchProbe;
	vector<pair<string, vector<path>>> matchGallery;
	for (auto&& probe : probeSigset) {
		string value = probe.identifier;
		vector<path> filepaths;
		for (auto&& pattern : probeMatchPatterns) {
			path fullFilePath = applyPattern(pattern, probe.dataPath);
			filepaths.push_back(fullFilePath);
		}
		matchProbe.push_back(make_pair(value, filepaths));
	}
	for (auto&& gallery : gallerySigset) {
		string value = gallery.identifier;
		vector<path> filepaths;
		for (auto&& pattern : galleryMatchPatterns) {
			path fullFilePath = applyPattern(pattern, gallery.dataPath);
			filepaths.push_back(fullFilePath);
		}
		matchGallery.push_back(make_pair(value, filepaths));
	}

	// For every probe, for every gallery, write a match-entry
	ptree matchlist;
	for (auto&& probe : matchProbe) {
		for (auto&& gallery : matchGallery) {
			ptree match;
			match.put("probe", probe.first);
			for (auto&& data : probe.second) {
				match.add("probe.path", data.string());
			}
			match.put("gallery", gallery.first);
			for (auto&& data : gallery.second) {
				match.add("gallery.path", data.string());
			}
			matchlist.add_child("match", match);
		}
	}
				
	boost::property_tree::write_info(matchlistFile.string(), matchlist);
	
	return 0;
}
