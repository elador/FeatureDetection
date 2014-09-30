/*
 * train-frame-extract-nnet.cpp
 *
 *  Created on: 11.09.2014
 *      Author: Patrik Huber
 *
 * Example:
 * train-frame-extract-nnet ...
 *   
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <iomanip>
#include <random>

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
#include "boost/lexical_cast.hpp"
#include "boost/archive/text_iarchive.hpp"

#include "tiny_cnn.h"

#include "facerecognition/pasc.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using cv::Mat;
using boost::filesystem::path;
using boost::lexical_cast;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::vector;
using std::string;

vector<Mat> getFrames(path videoFilename)
{
	vector<Mat> frames;

	cv::VideoCapture cap(videoFilename.string());
	if (!cap.isOpened())
		throw("Couldn't open video file.");

	Mat img;
	while (cap.read(img)) {
		frames.emplace_back(img);
	}

	return frames;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path inputDirectory;
	path inputLandmarks;
	path outputPath;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("input,i", po::value<path>(&inputDirectory)->required(),
				"input folder with training videos")
			("landmarks,l", po::value<path>(&inputLandmarks)->required(),
				"input landmarks")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: train-frame-extract-nnet [options]\n";
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
	Loggers->getLogger("train-frame-extract-nnet").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("train-frame-extract-nnet");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(inputLandmarks.string()); // ("pasc.bin", std::ios::binary | std::ios::in)
		boost::archive::text_iarchive ia(ifs); // binary_iarchive
		ia >> pascVideoDetections;
	} // archive and stream closed when destructors are called

	// Create the output directory if it doesn't exist yet:
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	
	// Read all videos:
	if (!boost::filesystem::exists(inputDirectory)) {
		cout << "hmm..";
	}
	vector<path> trainingVideos;
	try {
		copy(boost::filesystem::directory_iterator(inputDirectory), boost::filesystem::directory_iterator(), back_inserter(trainingVideos));
	}
	catch (boost::filesystem::filesystem_error& e) {
		// Log: Something went wrong while loading the videos. Details:
		cout << e.what();
	}
	std::random_device rd;
	auto videosSeed = rd();
	auto framesSeed = rd();
	std::mt19937 rndGenVideos(videosSeed);
	std::mt19937 rndGenFrames(framesSeed);
	std::uniform_int_distribution<> rndVidDistr(0, trainingVideos.size() - 1);
	auto randomVideo = std::bind(rndVidDistr, rndGenVideos);
	
	// The training data:
	vector<Mat> trainingFrames;
	vector<float> labels; // the score difference to the value we would optimally like
						  // I.e. if it's a positive pair, the label is the difference to 1.0
						  // In case of a negative pair, the label is the difference to 0.0

	// Select random subset of videos:
	int numVideosToTrain = 2;
	int numFramesPerVideo = 2;
	for (int i = 0; i < numVideosToTrain; ++i) {
		auto videoFilename = trainingVideos[randomVideo()];
		auto frames = getFrames(videoFilename);
		// Select random subset of frames:
		std::uniform_int_distribution<> rndFrameDistr(0, frames.size() - 1);
		for (int j = 0; j < numFramesPerVideo; ++j) {
			int frameNum = rndFrameDistr(rndGenFrames);
			auto frame = frames[frameNum];
			// Get the landmarks for this frame:
			frameNum = frameNum + 1; // frame numbering in the CSV starts with 1
			std::ostringstream ss;
			ss << std::setw(3) << std::setfill('0') << frameNum;
			string frameName = videoFilename.stem().string() + "/" + videoFilename.stem().string() + "-" + ss.str() + ".jpg";
			
			auto landmarks = std::find_if(begin(pascVideoDetections), end(pascVideoDetections), [frameName](const facerecognition::PascVideoDetection& d) { return (d.frame_id == frameName); });
			// Use facebox (later: or eyes) to run the engine
			if (landmarks == std::end(pascVideoDetections)) {
				continue; // instead, do a while-loop and count the number of frames with landmarks (so we don't skip videos if we draw bad values)
			}

			// Run the engine:
			// in: frame, eye/face coords, plus one positive image from the still-sigset with its eye/face coords
			// out: score
			// Later: Include several positive scores, and also negative pairs
			// I.e. enroll the whole gallery once, then match the query frame and get all scores?

			// From this pair:
			// resulting score difference to class label = label, facebox = input, resize it
			trainingFrames.emplace_back(frame);
			labels.emplace_back(1.0f);
		}
	}

	// Engine:
	// libFaceRecognition: CMake-Option for dependency on FaceVACS
	// Class Engine;
	// FaceVacsExecutableRunner : Engine; C'tor: Path to directory with binaries or individual binaries?
	// FaceVacs : Engine; C'tor - nothing? FV-Config?

	// Or: All the FaceVacs stuff in libFaceVacsWrapper. Direct-calls and exe-runner. With CMake-Option.
	
	
	// Train NN:
	// trainingFrames, labels


	// Save it:

	// Test it:
	// - Error on train-set?
	// - Split the train set in train/test
	// - Error on 'real' PaSC part? (i.e. the validation set)

	return EXIT_SUCCESS;
}
