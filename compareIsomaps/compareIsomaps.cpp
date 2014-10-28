/*
 * compareIsomaps.cpp
 *
 *  Created on: 12.06.2014
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

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <exception>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using cv::Mat;
using boost::filesystem::path;
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
			cout << "Usage: compareIsomaps [options]\n";
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

	Loggers->getLogger("compareIsomaps").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("compareIsomaps");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	path testList(R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\lists\probe_p15.txt)"); // only the basename gets used
	path groundtruthList(R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\lists\gallery_frontal.txt)"); // only the basename gets used

	path testIsomaps(R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\02_27062014_TexExtr_visCheck\fittings\probe_p15\)");
	path groundtruthIsomaps(R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\02_27062014_TexExtr_visCheck\fittings\gallery_frontal\)");

	std::ifstream testFile(testList.string());
	std::ifstream groundtruthFile(groundtruthList.string());

	vector<path> testImages;
	vector<path> groundtruthImages;
	
	// Load the images:
	std::string line;
	if (!testFile.is_open()) {
		return EXIT_FAILURE;
	}
	while (std::getline(testFile, line))
	{
		testImages.emplace_back(line);
	}
	if (!groundtruthFile.is_open()) {
		return EXIT_FAILURE;
	}
	while (std::getline(groundtruthFile, line))
	{
		groundtruthImages.emplace_back(line);
	}
	testFile.close();
	groundtruthFile.close();

	if (testImages.size() != groundtruthImages.size()) {
		return EXIT_FAILURE;
	}

	double totalNorm = 0.0;
	// Compare each image:
	for (auto i = 0; i < testImages.size(); ++i) {
		string testImageFile = testImages[i].stem().string();
		testImageFile += "_isomap.png";
		Mat testIsomap = cv::imread((testIsomaps / testImageFile).string());

		string groundtruthImageFile = groundtruthImages[i].stem().string();
		groundtruthImageFile += "_isomap.png";
		Mat groundtruthIsomap = cv::imread((groundtruthIsomaps / groundtruthImageFile).string());

		testIsomap.convertTo(testIsomap, CV_32FC3);
		groundtruthIsomap.convertTo(groundtruthIsomap, CV_32FC3);

		//Mat diff = testIsomap - groundtruthIsomap;
		double norm = cv::norm(testIsomap, groundtruthIsomap, cv::NORM_L2);
		// The channels get combined as follows: The norms of each individual channel is
		// computed. Afterwards, norm_total = sqrt(normR^2 + normG^2 + normB^2).
		totalNorm += norm;
	}
	totalNorm = totalNorm / testImages.size();

	appLogger.info("Total difference norm over all channels, averaged over all images: " + std::to_string(totalNorm));

	return 0;
}
