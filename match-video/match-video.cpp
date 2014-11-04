/*
 * match-video.cpp
 *
 *  Created on: 11.09.2014
 *      Author: Patrik Huber
 *
 * Ideally we'd use video, match against highres stills? (and not the lowres). Because if still are lowres/bad, we could match a
 * good frame against a bad gallery, which would give a bad score, but it shouldn't, because the frame is good.
 * Do we have labels for this?
 * Maybe "sensor_id","stage_id","env_id","illuminant_id" in the files emailed by Ross.
 *
 * Example:
 * match-video ...
 *   
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
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
#include "boost/archive/text_oarchive.hpp"
//#include "boost/archive/binary_iarchive.hpp"
//#include "boost/archive/binary_oarchive.hpp"

#include <boost/timer.hpp>
#include <boost/progress.hpp>

#include "imageio/MatSerialization.hpp"
#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"
#include "facerecognition/FaceVacsEngine.hpp"

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

class VideoScore
{
public:
	std::vector<std::vector<float>> scores; // For every frame, have a vector with the scores for all gallery
	// Gallery records?
private:

};

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path inputDirectoryVideos, inputDirectoryStills;
	path inputLandmarks;
	path outputPath;
	path fvsdkConfig;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("input-videos,i", po::value<path>(&inputDirectoryVideos)->required(),
				"input folder with training videos")
			("input-stills,j", po::value<path>(&inputDirectoryStills)->required(),
				"input folder with training videos")
			("landmarks,l", po::value<path>(&inputLandmarks)->required(),
				"input landmarks in boost::serialization text format")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
			("fvsdk-config,c", po::value<path>(&fvsdkConfig)->default_value(R"(C:\FVSDK_8_9_5\etc\frsdk.cfg)"),
				"path to frsdk.cfg. Usually something like C:\\FVSDK_8_9_5\\etc\\frsdk.cfg")
		;
		
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: match-video [options]" << std::endl;
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
	Loggers->getLogger("match-video").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("match-video");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	if (!boost::filesystem::exists(inputDirectoryVideos)) {
		appLogger.error("The given input video directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}
	if (!boost::filesystem::exists(inputDirectoryStills)) {
		appLogger.error("The given input stills directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}

	// Read the video detections metadata (eyes, face-coords):
	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(inputLandmarks.string()); // ("pasc.bin", std::ios::binary | std::ios::in)
		boost::archive::text_iarchive ia(ifs); // binary_iarchive
		ia >> pascVideoDetections;
	} // archive and stream closed when destructors are called

	// We would read the still images detection metadata (eyes, face-coords) here, but it's not provided with PaSC.
	// Todo: Try with and without the 5 Cog LMs

	// Read the training-video xml sigset and the training-still sigset to get the subject-id metadata:
	auto videoQuerySet = facerecognition::utils::readPascSigset(R"(C:\Users\Patrik\Documents\GitHub\data\PaSC\Web\nd1Fall2010VideoPaSCTrainingSet.xml)", true);
	auto stillTargetSet = facerecognition::utils::readPascSigset(R"(C:\Users\Patrik\Documents\GitHub\data\PaSC\Web\nd1Fall2010PaSCamerasStillTrainingSet.xml)", true);

	// Create the output directory if it doesn't exist yet:
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	
	path fvsdkTempDir{ "./tmp_fvsdk" };
	if (!boost::filesystem::exists(fvsdkTempDir)) {
		boost::filesystem::create_directory(fvsdkTempDir);
	}
	facerecognition::FaceVacsEngine faceRecEngine(fvsdkConfig, fvsdkTempDir);
	stillTargetSet.resize(1000); // 1000 = FIR limit atm
	faceRecEngine.enrollGallery(stillTargetSet, inputDirectoryStills);

	for (auto& video : videoQuerySet)
	{
		auto videoName = inputDirectoryVideos / video.dataPath;
		auto frames = facerecognition::utils::getFrames(videoName);
		VideoScore videoScore;

		path scoreOutputFile{ outputPath / videoName.filename().stem() };
		scoreOutputFile.replace_extension(".txt");
		std::ofstream out(scoreOutputFile.string());

		for (size_t frameNum = 0; frameNum < frames.size(); ++frameNum)
		{
			appLogger.debug("Processing frame " + std::to_string(frameNum));
			string frameName = facerecognition::getPascFrameName(video.dataPath, frameNum + 1); // 184 is a good test-video and it's in the first 100 gallery
			auto recognitionScores = faceRecEngine.matchAll(frames[frameNum], frameName, cv::Vec2f(), cv::Vec2f());

			// For the currently selected video, partition the target set. The distributions don't change each frame, whole video has the same FaceRecord.
			auto querySubject = video.subjectId;
			auto bound = std::partition(begin(recognitionScores), end(recognitionScores), [querySubject](std::pair<facerecognition::FaceRecord, float>& target) { return target.first.subjectId == querySubject; });
			// begin to bound = positive pairs, rest = negative
			auto numPositivePairs = std::distance(begin(recognitionScores), bound);
			auto numNegativePairs = std::distance(bound, end(recognitionScores));

			out << numPositivePairs << ",";
			out << numNegativePairs << ",";

			for (auto iter = begin(recognitionScores); iter != bound; ++iter) {
				out << iter->first.dataPath.stem().string() << "," << iter->second << ",";
			}
			for (auto iter = bound; iter != end(recognitionScores); ++iter) {
				out << iter->first.dataPath.stem().string() << "," << iter->second << ",";
			}
			out << endl;
		}
		out.close();
	}

	return EXIT_SUCCESS;
}
