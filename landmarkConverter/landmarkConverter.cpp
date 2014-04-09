/*
 * landmarkConverter.cpp
 *
 *  Created on: 04.04.2014
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

#include <memory>
#include <iostream>

#include "opencv2/core/core.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/lexical_cast.hpp"

#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/RectLandmark.hpp"
#include "imageio/IbugLandmarkFormatParser.hpp"
#include "imageio/MuctLandmarkFormatParser.hpp"
#include "imageio/DidLandmarkSink.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using boost::property_tree::ptree;
using boost::filesystem::path;
using boost::lexical_cast;
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
	path inputLandmarks;
	path outputLandmarks;
	string inputLandmarkType;
	string outputLandmarkType;
	path landmarkMappingsFile;
	
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("input,i", po::value<path>(&inputLandmarks)->required(), 
				"input landmarks file or folder")
			("input-type,s", po::value<string>(&inputLandmarkType)->required(), 
				"type of input landmarks")
			("output,o", po::value<path>(&outputLandmarks)->required(), 
				"output folder")
			("output-type,t", po::value<string>(&outputLandmarkType)->required(),
				"type of output landmarks")
			("mapping,m", po::value<path>(&landmarkMappingsFile)->required(),
				"a file with mappings from the input- to the output-format")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: landmarkConverter [options]\n";
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
		return EXIT_SUCCESS;
	}
	
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("landmarkConverter").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("landmarkConverter");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Load the input landmarks
	shared_ptr<NamedLandmarkSource> landmarkSource;
	vector<path> groundtruthDirs; groundtruthDirs.push_back(inputLandmarks); // Todo: Make cmdline use a vector<path>
	shared_ptr<LandmarkFormatParser> landmarkFormatParser;
	if (boost::iequals(inputLandmarkType, "muct76-opencv")) {
		landmarkFormatParser = make_shared<MuctLandmarkFormatParser>();
		landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(shared_ptr<ImageSource>(), string(), GatherMethod::SEPARATE_FILES, groundtruthDirs), landmarkFormatParser);
	}
	//else if (boost::iequals(inputLandmarkType, "ibug")) {
	//	landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
	//}
	else {
		appLogger.error("The input landmark type is not supported.");
		return EXIT_SUCCESS;
	}

	// Load the mapping file
	map<string, string> landmarkMappings;
	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(landmarkMappingsFile.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	try { // TODO: Make a LandmarkMapping class and put this into (c'tor ptree, return map) or load()
		ptree ptLandmarkMappings = pt.get_child("landmarkMappings");
		for (const auto& mapping : ptLandmarkMappings) {
			landmarkMappings.insert(make_pair(mapping.first, mapping.second.get_value<string>()));
		}
		appLogger.info("Loaded a list of " + lexical_cast<string>(landmarkMappings.size()) + " landmark mappings.");
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(string("Error while parsing the mappings file: ") + error.what());
		return EXIT_FAILURE;
	}
	catch (const std::runtime_error& error) {
		appLogger.error(string("Error while parsing the mappings file: ") + error.what());
		return EXIT_FAILURE;
	}

	// Create the landmark-sink for the output landmarks
	shared_ptr<NamedLandmarkSink> landmarkSink;
	if (boost::iequals(outputLandmarkType, "did")) {
		landmarkSink = make_shared<DidLandmarkSink>();
	}
	//else if (boost::iequals(outputLandmarkType, "ibug")) {
	//	landmarkSink = make_shared<DidLandmarkSink>();
	//}
	else {
		appLogger.error("The output landmark type is not supported.");
		return EXIT_SUCCESS;
	}

	// Create the output directory
	if (!boost::filesystem::exists(outputLandmarks)) {
		boost::filesystem::create_directory(outputLandmarks);
	}

	while(landmarkSource->next()) {
		appLogger.info("Converting " + landmarkSource->getName().string());
		LandmarkCollection lms = landmarkSource->getLandmarks();
		const auto& oldLandmarks = lms.getLandmarks();
		LandmarkCollection convertedLandmarks;
		for (const auto& old : oldLandmarks) {
			if (landmarkMappings.find(old->getName()) != end(landmarkMappings))	{
				std::string mappedId = landmarkMappings.at(old->getName());
				shared_ptr<Landmark> convertedLandmark;
				switch (old->getType())
				{
				case Landmark::LandmarkType::MODEL:
					convertedLandmark = make_shared<ModelLandmark>(mappedId, old->getPosition3D(), old->isVisible());
					break;
				case Landmark::LandmarkType::RECT:
					convertedLandmark = make_shared<RectLandmark>(mappedId, old->getPosition2D(), old->getSize(), old->isVisible());
					break;
				default:
					appLogger.error("Encountered an unknown LandmarkType. Please update this switch-statement.");
					throw std::runtime_error("Encountered an unknown LandmarkType. Please update this switch-statement.");
					break;
				}
				convertedLandmarks.insert(convertedLandmark);
			}
			else {
				appLogger.info("Could not find the current landmark \"" + old->getName() + "\" in the landmarks-mapping, not writing this landmark into the output file.");
			}
		}
		
		path outputFilename = outputLandmarks / landmarkSource->getName().stem();
		landmarkSink->add(convertedLandmarks, outputFilename);
	}
	appLogger.info("Finished converting all landmarks.");

	return 0;
}
