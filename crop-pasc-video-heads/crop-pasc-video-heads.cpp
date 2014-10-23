/*
 * crop-pasc-video-heads.cpp
 *
 *  Created on: 23.10.2014
 *      Author: Patrik Huber
 *
 * Example:
 * crop-pasc-video-heads ...
 *   
 */

#include <memory>
#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem.hpp"
#include "boost/archive/text_iarchive.hpp"

#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using boost::filesystem::path;
using cv::Mat;
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
	path sigsetFile, metadataFile, inputDirectory;
	path outputFolder;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
//			("sigset,s", po::value<path>(&sigsetFile)->required(),
//				"PaSC video XML sigset of the frames to be cropped")
			("input,i", po::value<path>(&inputDirectory)->required(),
				"directory containing the frames. Files should be in the format 'videoname.012.png'")
			("metadata,m", po::value<path>(&metadataFile)->required(),
				"PaSC video detections metadata in boost::serialization format")
			("output,o", po::value<path>(&outputFolder)->default_value("."),
				"path to save the cropped patches to")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: crop-pasc-video-heads [options]" << endl;
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

	if (!fs::exists(inputDirectory) || fs::is_regular_file(inputDirectory)) {
		return EXIT_FAILURE;
	}
	
	// Create the output directory if it doesn't exist yet
	if (!boost::filesystem::exists(outputFolder)) {
		boost::filesystem::create_directory(outputFolder);
	}

	

	// Read the video detections metadata (eyes, face-coords):
	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(metadataFile.string());
		boost::archive::text_iarchive ia(ifs);
		ia >> pascVideoDetections;
	}

	//auto sigset = facerecognition::utils::readPascSigset(sigsetFile, true);
	vector<path> files;
	try	{
		std::copy(fs::directory_iterator(inputDirectory), fs::directory_iterator(), std::back_inserter(files));
	}
	catch (const fs::filesystem_error& ex)
	{
		cout << ex.what() << endl;
	}

	for (auto& file : files) {
		appLogger.info("Processing " + file.string());
		if (!fs::is_regular(file) || file.extension() != ".png") {
			appLogger.debug("Not a regular file or no .png extension, skipping: " + file.string());
			continue;
		}
		string frameNumExtension = file.stem().extension().string(); // from videoname.123.png to ".123"
		frameNumExtension.erase(std::remove(frameNumExtension.begin(), frameNumExtension.end(), '.'), frameNumExtension.end());
		int frameNum = boost::lexical_cast<int>(frameNumExtension);
		path videoName = file.stem().stem(); // from videoname.123.png to "videoname"
		videoName.replace_extension(".mp4");
		string frameName = facerecognition::getPascFrameName(videoName, frameNum);
		auto landmarks = std::find_if(begin(pascVideoDetections), end(pascVideoDetections), [frameName](const facerecognition::PascVideoDetection& d) { return (d.frame_id == frameName); });
		if (landmarks == end(pascVideoDetections)) {
			string logMessage("Frame has no PittPatt detections in the metadata file. This shouldn't happen, or rather, we do not want it to happen, because we didn't select a frame where this should happen.");
			appLogger.error(logMessage);
			throw std::runtime_error(logMessage);
		}
		int tlx = landmarks->fcen_x - landmarks->fwidth / 2.0;
		int tly = landmarks->fcen_y - landmarks->fheight / 2.0;
		int w = landmarks->fwidth;
		int h = landmarks->fheight;
		Mat frame = cv::imread(file.string());
		if (tlx < 0 || tlx + w >= frame.cols || tly < 0 || tly + h >= frame.rows) {
			// patch has some regions outside the image
			string logMessage("Throwing away patch because it goes outside the image bounds. This shouldn't happen, or rather, we do not want it to happen, because we didn't select a frame where this should happen.");
			appLogger.error(logMessage);
			throw std::runtime_error(logMessage);
		}
		cv::Rect roi(tlx, tly, w, h);
		Mat croppedFace = frame(roi);
		path croppedImageFilename = outputFolder / file.filename();
		cv::imwrite(croppedImageFilename.string(), croppedFace);
	}

	return EXIT_SUCCESS;
}
