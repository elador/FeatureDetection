/*
 * generateMultipieList.cpp
 *
 *  Created on: 10.06.2014
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

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"

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

	Loggers->getLogger("generateMultipieList").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("generateMultipieList");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	path outputImageList(R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\lists\probe_p15.txt)");

	//path multipieRoot(R"(Z:\datasets\still01\multiPIE\data\)");
	path multipieRoot(R"(K:\still01\multiPIE\data\)");
	// MULTIPIE_ROOT_DIR/session0x/multiview/subjId/exprNum/camera/subj_session_expr_cam_imgNum.png
	// Example: MULTIPIE_ROOT_DIR / 001 / 01 / 01_0 / 001_01_01_010_00.png
	path session("session"); // add 01, 02, 03 or 04
	path type("multiview"); // highres, movies or multiview - only multiview supported at the moment

	vector<string> subjects{ }; // an empty vector means use all - only empty supported at the moment
	vector<string> sessions{ "01" };
	vector<string> recordingIds{ "01" }; // Not unique. E.g. in session01, id02 is smile, while in session02, id02 is surprise... So: Only enter 1 session + multiple recording Ids, OR, multiple sessions and 1 recording Id.
	
	//vector<string> cameras{ "09_0", "20_0", "08_0", "19_0", "13_0", "04_1", "14_0", "05_0" }; // probes - +-60, 45, 30, 15 yaw angle
	vector<string> cameras{ "05_0" }; // 
	//vector<string> cameras{ "05_1" }; // gallery - frontal
	vector<string> lighting{ "07" }; // probes
	//vector<string> lighting{ "07" }; // gallery

	std::ofstream filelist(outputImageList.string());
	
	for (auto&& sess : sessions) {
		// Build the subject list
		vector<string> subjectIds;
		if (subjects.empty()) { // an empty vector means use all - only empty supported at the moment
			path fullsession = session;
			fullsession += sess;
			path subjectsDir = multipieRoot / fullsession / type;
			if (!fs::exists(subjectsDir))
				throw std::runtime_error("Directory '" + subjectsDir.string() + "' does not exist.");
			if (!fs::is_directory(subjectsDir))
				throw std::runtime_error("'" + subjectsDir.string() + "' is not a directory.");
			vector<path> subjectDirectories;
			std::copy(fs::directory_iterator(subjectsDir), fs::directory_iterator(), std::back_inserter(subjectDirectories));
			for (auto&& dir : subjectDirectories) {
				if (fs::is_directory(dir))
					subjectIds.push_back(dir.filename().string());
			}
		}
		else {
			subjectIds = subjects;
		}
		// Collect the files
		for (auto&& subject : subjectIds) {
			for (auto&& recording : recordingIds) {
				for (auto&& camera : cameras) {
					for (auto&& light : lighting) {
						string cam = camera;
						cam.erase(std::remove(begin(cam), end(cam), '_'), end(cam)); // remove the "_" to form the filename
						string filename = subject + "_" + sess + "_" + recording + "_" + cam + "_" + light + ".png";
						path fullsession = session;
						fullsession += sess;
						path fullFilePath = multipieRoot / fullsession / type / subject / recording / camera / filename;
						filelist << fullFilePath.string() << endl;
					}
				}
			}
		}
	}

	filelist.close();
	return 0;
}
