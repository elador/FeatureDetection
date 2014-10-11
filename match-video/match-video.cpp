/*
 * match-video.cpp
 *
 *  Created on: 11.10.2014
 *      Author: Patrik Huber
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

#include <frsdk/config.h>
#include <frsdk/enroll.h>
#include <frsdk/match.h>
#include <frsdk/eyes.h>


#include <string>
#include <sstream>
#include <frsdk/image.h>
#include <frsdk/jpeg.h>

#include "imageio/MatSerialization.hpp"
#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"

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
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: train-frameselect-extract [options]" << std::endl;
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

	// Read the video detections metadata (eyes, face-coords):
	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(inputLandmarks.string()); // ("pasc.bin", std::ios::binary | std::ios::in)
		boost::archive::text_iarchive ia(ifs); // binary_iarchive
		ia >> pascVideoDetections;
	} // archive and stream closed when destructors are called

	// We would read the still images detection metadata (eyes, face-coords) here, but it's not provided with PaSC. Emailed Ross.
	// TODO!

	// Read the training-video xml sigset and the training-still sigset to get the subject-id metadata:
	auto videoQuerySet = facerecognition::utils::readPascSigset(R"(C:\Users\Patrik\Documents\GitHub\data\PaSC\Web\nd1Fall2010VideoPaSCTrainingSet.xml)", true);
	auto stillTargetSet = facerecognition::utils::readPascSigset(R"(C:\Users\Patrik\Documents\GitHub\data\PaSC\Web\nd1Fall2010PaSCamerasStillTrainingSet.xml)", true);

	// Create the output directory if it doesn't exist yet:
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	
	// Read all videos:
	if (!boost::filesystem::exists(inputDirectoryVideos)) {
		appLogger.error("The given input files directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}

	auto stillTrainingGBU = facerecognition::utils::readPascSigset(R"(C:\Users\Patrik\Documents\GitHub\data\PaSC\Mail_Ross\GBU_Training_Uncontrolledx8.xml)");
	vector<path> trainingStills;

	cout << "Stills over files:" << endl;
	int numVideosThatHaveAnIdInStillGBU = 0;
	for (auto& s : videoQuerySet) {
		auto res = std::find_if(begin(stillTrainingGBU), end(stillTrainingGBU), [s](const facerecognition::FaceRecord& fr) { return (fr.subjectId == s.subjectId); });
		if (res == std::end(stillTrainingGBU)) {
			
			//cout << "I'm on the HDD, but not in the XML: " << fn << endl;
		}
		else {
			++numVideosThatHaveAnIdInStillGBU;
		}
	}
	cout << "Stills over xml:" << endl;
	int numStillOfXmlThatExistAsFiles = 0;
	for (auto& s : stillTrainingGBU) {
		auto p = inputDirectoryStills / s.dataPath;
		if (boost::filesystem::exists(p)) {
			++numStillOfXmlThatExistAsFiles;
		}
		else {
			//cout << "I'm in the XML, but not in the Dir: " << p.string() << endl;
		}
	}

	/*
	vector<path> trainingVideos;
	try {
		copy(boost::filesystem::directory_iterator(inputDirectoryVideos), boost::filesystem::directory_iterator(), back_inserter(trainingVideos));
	}
	catch (boost::filesystem::filesystem_error& e) {
		string errorMsg("Error while loading the video files from the given input directory: " + string(e.what()));
		appLogger.error(errorMsg);
		return EXIT_FAILURE;
	}*/

	FaceVacsEngine faceRecEngine(R"(C:\FVSDK_8_9_5\etc\frsdk.cfg)", R"(C:\Users\Patrik\Documents\GitHub\aaatmp)");
	//auto stillTargetSetSmall = { stillTargetSet[1092]/*, stillTargetSet[0] */};
	//path = L"C:\\Users\\Patrik\\Documents\\GitHub\\aaatmp\\06340d96.fir"
	//auto dp = path("06340d96.jpg");
	//auto galSubj = std::find_if(begin(stillTargetSet), end(stillTargetSet), [dp](const facerecognition::FaceRecord& g) { return (g.dataPath == dp); });
	// check
	//auto galSubjIdx = std::distance(begin(stillTargetSet), galSubj);
	stillTargetSet.resize(1000); // 1000 = FIR limit atm
	faceRecEngine.enrollGallery(stillTargetSet, inputDirectoryStills);

	std::random_device rd;
	auto videosSeed = rd();
	auto framesSeed = rd();
	auto posTargetsSeed = rd();
	auto negTargetsSeed = rd();
	std::mt19937 rndGenVideos(videosSeed);
	std::mt19937 rndGenFrames(framesSeed);
	std::mt19937 rndGenPosTargets(posTargetsSeed);
	std::mt19937 rndGenNegTargets(negTargetsSeed);
	std::uniform_int_distribution<> rndVidDistr(0, videoQuerySet.size() - 1);
	auto randomVideo = std::bind(rndVidDistr, rndGenVideos);
	
	// The training data:
	vector<Mat> trainingFrames;
	vector<float> labels; // the score difference to the value we would optimally like
						  // I.e. if it's a positive pair, the label is the difference to 1.0
						  // In case of a negative pair, the label is the difference to 0.0

	// Select random subset of videos: (Note: This means we may select the same video twice - not so optimal?)
	int numVideosToTrain = 60;
	int numFramesPerVideo = 80;
	int numPositivePairsPerFrame = 2;
	int numNegativePairsPerFrame = 0;
	for (int i = 0; i < numVideosToTrain; ++i) {
		numPositivePairsPerFrame = 1; // it has to be set to 1 again if we set it to 0 below in the previous iteration.
		auto queryVideo = videoQuerySet[randomVideo()];
		if (!boost::filesystem::exists(inputDirectoryVideos / queryVideo.dataPath)) {  // Shouldn't be necessary, but there are 5 videos in the xml sigset that we don't have.
			continue;
		}
		auto frames = getFrames(inputDirectoryVideos / queryVideo.dataPath);
		// For the currently selected video, partition the target set. The distributions don't change each frame, whole video has the same FaceRecord.
		auto bound = std::partition(begin(stillTargetSet), end(stillTargetSet), [queryVideo](facerecognition::FaceRecord& target) { return target.subjectId == queryVideo.subjectId; });
		// begin to bound = positive pairs, rest = negative
		auto numPositivePairs = std::distance(begin(stillTargetSet), bound);
		auto numNegativePairs = std::distance(bound, end(stillTargetSet));
		if (numPositivePairs == 0) { // Not sure if this should be happening in the PaSC sigsets - ask Ross
			numPositivePairs = 1; // Causes the uniform_int_distribution c'tor not to crash because of negative value
			numPositivePairsPerFrame = 0; // Causes the positive-pairs stuff further down to be skipped
			continue; // only for debugging!
		}
		std::uniform_int_distribution<> rndPositiveDistr(0, numPositivePairs - 1); // -1 because all vector indices start with 0
		std::uniform_int_distribution<> rndNegativeDistr(numPositivePairs, stillTargetSet.size() - 1);
		// Select random subset of frames:
		std::uniform_int_distribution<> rndFrameDistr(0, frames.size() - 1);
		for (int j = 0; j < numFramesPerVideo; ++j) {
			int frameNum = rndFrameDistr(rndGenFrames);
			auto frame = frames[frameNum];
			// Get the landmarks for this frame:
			string frameName = getPascFrameName(queryVideo.dataPath, frameNum + 1);
			std::cout << "=== STARTING TO PROCESS " << frameName << " ===" << std::endl;
			auto landmarks = std::find_if(begin(pascVideoDetections), end(pascVideoDetections), [frameName](const facerecognition::PascVideoDetection& d) { return (d.frame_id == frameName); });
			// Use facebox (later: or eyes) to run the engine
			if (landmarks == std::end(pascVideoDetections)) {
				appLogger.info("Chose a frame but could not find a corresponding entry in the metadata file - skipping it.");
				continue; // instead, do a while-loop and count the number of frames with landmarks (so we don't skip videos if we draw bad values)
				// We throw away the frames with no landmarks. This basically means our algorithm will only be trained on frames where PittPatt succeeds, and
				// frames where it doesn't are unknown data to our nnet. I think we should try including these frames as well, e.g. with an error/label of 1.0.
			}
			// Choose one random positive and one random negative pair (we could use more, this is just the first attempt):
			// Actually with the Cog engine we could enrol the whole gallery and get all the scores in one go, should be much faster
			for (int k = 0; k < numPositivePairsPerFrame; ++k) { // we can also get twice (or more) times the same, but doesn't matter for now
				auto targetStill = stillTargetSet[rndPositiveDistr(rndGenPosTargets)];
				// TODO get LMs for targetStill from PaSC - see further up, email Ross
				// match (targetStill (FaceRecord), LMS TODO ) against ('frame' (Mat), queryVideo (FaceRecord), landmarks)
				double recognitionScore = faceRecEngine.match(inputDirectoryStills / targetStill.dataPath, targetStill.subjectId, frame, frameName, cv::Vec2f(), cv::Vec2f());
				std::cout << "===== Got a score!" << recognitionScore << "=====" << std::endl;
				string out = (outputPath / path(frameName).stem()).string() + "_" + std::to_string(recognitionScore) + ".png";
				cv::imwrite(out, frame);
				double frameScore = recognitionScore; // This is our label. It's a positive pair, so the higher the score the higher our framescore.

				// The face box is always given. Only the eyes are missing sometimes.
				int tlx = landmarks->fcen_x - landmarks->fwidth / 2.0 - landmarks->fwidth / 10.0; // go a little further than the box-width
				int tly = landmarks->fcen_y - landmarks->fheight / 2.0 - landmarks->fheight / 10.0;
				int w = landmarks->fwidth + landmarks->fwidth / 5.0;
				int h = landmarks->fheight + landmarks->fheight / 5.0;
				if (tlx < 0 || tlx + w >= frame.cols || tly < 0 || tly + h >= frame.rows) {
					// patch has some regions outside the image
					continue;
				}
				cv::Rect roi(tlx, tly, w, h);
				Mat croppedFace = frame(roi);
				cv::resize(croppedFace, croppedFace, cv::Size(32, 32)); // need to set this higher... 40x40? What about aspect ratio?
				cv::cvtColor(croppedFace, croppedFace, cv::COLOR_BGR2GRAY);

				trainingFrames.emplace_back(croppedFace);
				labels.emplace_back(frameScore);
			}
			for (int k = 0; k < numNegativePairsPerFrame; ++k) {
				double recognitionScore;
				double frameScore = 1.0 - recognitionScore; // This is our label. It's a negative pair, so a facerec-score close to 0 is good, meaning 1-frScore is the frameScore
				// This is problematic here because we return a score of 0 for frames that we couldn't enrol - they would get 0 error here, which is bad.
			}

			// Run the engine:
			// in: frame, eye/face coords, plus one positive image from the still-sigset with its eye/face coords
			// out: score
			// Later: Include several positive scores, and also negative pairs
			// I.e. enroll the whole gallery once, then match the query frame and get all scores?

			// From this pair:
			// resulting score difference to class label = label, facebox = input, resize it
			//trainingFrames.emplace_back(frame); // store only the resized frames!
			//labels.emplace_back(1.0f); // 1.0 means good frame, 0.0 means bad frame.
		}
	}

	// Engine:
	// libFaceRecognition: CMake-Option for dependency on FaceVACS
	// Class Engine;
	// FaceVacsExecutableRunner : Engine; C'tor: Path to directory with binaries or individual binaries?
	// FaceVacs : Engine; C'tor - nothing? FV-Config?

	// Or: All the FaceVacs stuff in libFaceVacsWrapper. Direct-calls and exe-runner. With CMake-Option.

	std::ofstream ofPascT("frames_data.txt");
	{ // use scope to ensure archive goes out of scope before stream
		boost::archive::text_oarchive oa(ofPascT);
		oa << trainingFrames;
	}
	ofPascT.close();

	std::ofstream ofPascT2("frames_labels.txt");
	{ // use scope to ensure archive goes out of scope before stream
		boost::archive::text_oarchive oa(ofPascT2);
		oa << labels;
	}
	ofPascT2.close();

	return EXIT_SUCCESS;
}
