/*
 * extract-frames.cpp
 *
 *  Created on: 29.09.2014
 *      Author: Patrik Huber
 *
 * Example:
 * extract-frames -i ...
 *   
 */

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
//#define _CRTDBG_MAP_ALLOC
#include <cstdlib>

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
#include <string>
#include <random>
#include <iomanip>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"

#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/PascVideoEyesLandmarkFormatParser.hpp"
#include "imageio/SimpleModelLandmarkSink.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using cv::Mat;
using boost::filesystem::path;
using std::string;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::vector;

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
	path inputFilename;
	//path inputLandmarks;
	//string landmarkType;
	//path landmarkMappings;
	path outputPath;
	string extractionMethod;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("input,i", po::value<path>(&inputFilename)->required(),
				"input video")
			//("landmarks,l", po::value<path>(&inputLandmarks)->required(),
			//	"input landmarks")
			//("landmark-type,t", po::value<string>(&landmarkType)->required(),
			//	"specify the type of landmarks: ibug")
			//("landmark-mappings,m", po::value<path>(&landmarkMappings),
			//	"an optional mapping-file that maps from the input landmarks to landmark identifiers in the model's format")
			("method,m", po::value<string>(&extractionMethod)->required(),
				"how to extract the frame(s): first, middle, random, maxFaceWidth")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: extract-frames [options]\n";
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
	
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("extract-frames").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("extract-frames");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Read the PaSC video landmarks:
	shared_ptr<imageio::NamedLandmarkSource> landmarkSource;
	vector<path> groundtruthDirs{ path(R"(C:\Users\Patrik\Documents\GitHub\data\PaSC\pasc_video_pittpatt_detections_debug.csv)") };
	shared_ptr<imageio::LandmarkFormatParser> landmarkFormatParser;
	string groundtruthType = "PaSC-still-PittPatt-eyes";
	if (boost::iequals(groundtruthType, "PaSC-still-PittPatt-eyes")) { // Todo/Note: Not sure this is working?
		landmarkFormatParser = make_shared<imageio::PascVideoEyesLandmarkFormatParser>();
		landmarkSource = make_shared<imageio::DefaultNamedLandmarkSource>(imageio::LandmarkFileGatherer::gather(nullptr, ".csv", imageio::GatherMethod::SEPARATE_FILES, groundtruthDirs), landmarkFormatParser);
	}
	else {
		appLogger.error("Invalid ground-truth landmarks type.");
		return EXIT_FAILURE;
	}

	// Output landmarks for the extracted frame:
	imageio::SimpleModelLandmarkSink landmarkSink;

	// Create the output directory if it doesn't exist yet
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}

	cv::VideoCapture cap(inputFilename.string());
	if (!cap.isOpened())
		return EXIT_FAILURE;

	appLogger.info("Extracting frame using method: " + extractionMethod);

	int frameToExtract;
	if (extractionMethod == "first" || extractionMethod == "middle" || extractionMethod == "random") {
		int frameCount = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_COUNT));
		if (frameCount == 0) {
			// error, property not supported.
			return EXIT_FAILURE;
		}
		if (extractionMethod == "first") {
			frameToExtract = 0;
		}
		else if (extractionMethod == "middle") {
			frameToExtract = frameCount / 2; // rounding down on odd numbers
		}
		else if (extractionMethod == "random") {
			std::uniform_int_distribution<> rndInt(0, frameCount - 1);
			std::mt19937 engine{};
			frameToExtract = rndInt(engine);
		}
	}
	
	if (extractionMethod == "maxFaceWidth") {
		Mat img;
		int currentFrame = 0;
		vector<double> interEyeDistances;
		while (cap.read(img)) {
			currentFrame++; // frame numbering in the CSV starts with 1
			std::ostringstream ss;
			ss << std::setw(3) << std::setfill('0') << currentFrame;
			string frameName = inputFilename.stem().string() + "/" + inputFilename.stem().string() + "-" + ss.str() + ".jpg";
			imageio::LandmarkCollection landmarks = landmarkSource->get(frameName);
			if (landmarks.isEmpty()) {
				interEyeDistances.emplace_back(0.0);
			}
			else {
				auto le = landmarks.getLandmark("le");
				auto re = landmarks.getLandmark("re");
				interEyeDistances.emplace_back(cv::norm(le->getPosition2D(), re->getPosition2D(), cv::NORM_L2));
			}
		}
		auto result = std::max_element(begin(interEyeDistances), end(interEyeDistances));
		frameToExtract = std::distance(begin(interEyeDistances), result);
	}

	// Get and write the frame image:
	Mat img;
	cap.set(CV_CAP_PROP_POS_FRAMES, frameToExtract); // starts at 0
	cap >> img;
	path fn = outputPath / inputFilename.filename();
	frameToExtract++; // PaSC starts naming them from 1
	fn.replace_extension(std::to_string(frameToExtract) + ".png");
	cv::imwrite(fn.string(), img);

	// Get and write the frames landmarks:
	std::ostringstream ss;
	ss << std::setw(3) << std::setfill('0') << frameToExtract;
	string frameName = inputFilename.stem().string() + "/" + inputFilename.stem().string() + "-" + ss.str() + ".jpg";
	imageio::LandmarkCollection landmarks;
	// we only want the eyes, not the face (not a ModelLandmark):
	landmarks.insert(landmarkSource->get(frameName).getLandmark("le"));
	landmarks.insert(landmarkSource->get(frameName).getLandmark("re"));
	fn.replace_extension(".txt");
	landmarkSink.add(landmarks, fn);

	appLogger.info("Finished extracting frame(s).");

	return EXIT_SUCCESS;
}
