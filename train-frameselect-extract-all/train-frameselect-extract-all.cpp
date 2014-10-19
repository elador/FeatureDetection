/*
 * train-frame-extract-nnet.cpp
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
 * train-frameselect-extract-all -q "C:\Users\Patrik\Documents\GitHub\data\PaSC\Web\nd1Fall2010VideoPaSCTrainingSet.xml" -r "Z:\datasets\multiview02\PaSC\training\video" -l "C:\Users\Patrik\Documents\GitHub\data\PaSC\pasc_training_video_pittpatt_detection.txt" -o out
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
#include "boost/serialization/utility.hpp"

#include <boost/timer.hpp>
#include <boost/progress.hpp>

#include "imageio/MatSerialization.hpp"
#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"
#include "facerecognition/FaceVacsEngine.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
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
	path inputDirectoryVideos, scoresDirectory;
	path querySigset;
	path queryLandmarks;
	path outputPath;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("query-sigset,q", po::value<path>(&querySigset)->required(),
				"PaSC video training query sigset")
			("query-path,r", po::value<path>(&inputDirectoryVideos)->required(),
				"path to the training videos")
			("scores,s", po::value<path>(&scoresDirectory)->required(),
				"path to a directory with face recognition scores in boost::serialization text format")
			("query-landmarks,l", po::value<path>(&queryLandmarks)->required(),
				"PaSC landmarks for the training videos in boost::serialization text format")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: train-frameselect-extract-all [options]" << std::endl;
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
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Read the video detections metadata (eyes, face-coords):
	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(queryLandmarks.string()); // ("pasc.bin", std::ios::binary | std::ios::in)
		boost::archive::text_iarchive ia(ifs); // binary_iarchive
		ia >> pascVideoDetections;
	} // archive and stream closed when destructors are called

	// We would read the still images detection metadata (eyes, face-coords) here, but it's not provided with PaSC. Emailed Ross.
	// TODO!

	// Read the training-video xml sigset and the training-still sigset to get the subject-id metadata:
	auto videoQuerySet = facerecognition::utils::readPascSigset(querySigset, true);

	// Create the output directory if it doesn't exist yet:
	if (!fs::exists(outputPath)) {
		fs::create_directory(outputPath);
	}
	
	// Read all videos:
	if (!fs::exists(inputDirectoryVideos)) {
		appLogger.error("The given input files directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}

	// The training data:
	vector<Mat> trainingFrames;
	vector<float> labels; // the score difference to the value we would optimally like
						  // I.e. if it's a positive pair, the label is the difference to 1.0
						  // In case of a negative pair, the label is the difference to 0.0

	//int numPositivePairsPerFrame = 2;
	int numNegativePairsPerFrame = 0;
	// Try: Out of pos (4-5), select max score

	for (auto& queryVideo : videoQuerySet) {
		// Shouldn't be necessary, but there are 5 videos in the xml sigset that we don't have:
		if (!fs::exists(inputDirectoryVideos / queryVideo.dataPath)) {
			continue;
		}
		// Get the saved scores:
		facerecognition::utils::VideoScore videoScores; // a vector<float> for every frame
		{
			path tmp = scoresDirectory / queryVideo.dataPath.stem();
			tmp.replace_extension(".bs.txt");
			if (!fs::exists(tmp)) {
				// Scores don't exist, this shouldn't happen, as we process all 280 PaSC-training videos
				string msg("Score file for this video does not exist. Did you specify the correct directory? " + tmp.string());
				appLogger.error(msg);
				throw std::runtime_error(msg);
			}
			std::ifstream ifs(tmp.string());
			boost::archive::text_iarchive ia(ifs);
			ia >> videoScores;
		} // archive and stream closed when destructors are called

		auto frames = facerecognition::utils::getFrames(inputDirectoryVideos / queryVideo.dataPath);

		for (int frameNum = 0; frameNum < frames.size(); ++frameNum) {
			auto& frame = frames[frameNum];
			float frameScore;

			// Get the PaSC PittPatt groundtruth landmarks for this frame:
			string frameName = facerecognition::getPascFrameName(queryVideo.dataPath, frameNum + 1);
			appLogger.debug("Starting to process " + frameName);
			auto landmarks = std::find_if(begin(pascVideoDetections), end(pascVideoDetections), [frameName](const facerecognition::PascVideoDetection& d) { return (d.frame_id == frameName); });
			if (landmarks == std::end(pascVideoDetections)) {
				appLogger.debug("Frame has no PittPatt detections in the metadata file.");
				//continue;
				// We throw away the frames with no landmarks. This basically means our algorithm will only be trained on frames where PittPatt succeeds, and
				// frames where it doesn't are unknown data to our nnet. I think we should try including these frames as well, e.g. with an error/label of 1.0.
				// ==> Bad frame, give bad label!
				frameScore = 0.0f;
				// ===> Well, the problem is, we can't use it (at least not the face patch), because we don't have eye/face coordinates...
				// We have some frames (how many?) where PittPatt succeeds but FaceVACS does not. This should give us a few very bad frames.
				continue;
			}

			// For the currently selected video, partition the target set. The distributions don't change each frame, whole video shows the same subject.
			// It would suffice to do this once per video. However, we'd need to adjust some stuff in the code...
			auto& recognitionScores = videoScores.scores[frameNum];
			if (recognitionScores.size() == 0) {
				// bad frame, give bad label!
				appLogger.debug("The engine couldn't produce scores for this frame, i.e. most likely couldn't enrol it.");
				frameScore = 0.0f;
			}
			auto bound = std::partition(begin(recognitionScores), end(recognitionScores), [queryVideo](std::pair<facerecognition::FaceRecord, float>& target) { return target.first.subjectId == queryVideo.subjectId; });
			// begin to bound = positive pairs, rest = negative
			auto numPositivePairs = std::distance(begin(recognitionScores), bound);
			auto numNegativePairs = std::distance(bound, end(recognitionScores));
			// numPositivePairs == 0:
			// For this subject in the video, we couldn't enrol any gallery subjects. It basically means the gallery images suck.
			// If anything, we could still use the negative scores.
			// With respect to PaSC, I think this shouldn't happen if our enrolment were error-less.

			float maxScore = 0.0f;
			int maxPos = 0;
			for (int k = 0; k < numPositivePairs; ++k) { // we can also get twice (or more) times the same, but doesn't matter for now
				// Max for now, but could use all
				if (recognitionScores[k].second >= maxScore) {
					maxScore = recognitionScores[k].second;
					maxPos = k;
				}
			}

			appLogger.debug("Maximum score amongst " + std::to_string(numPositivePairs) + " positive pairs: " + std::to_string(maxScore));
			frameScore = maxScore; // This is our label. It's a positive pair, so the higher the score the higher our framescore.

			// The face box is always given. Only the eyes are missing sometimes.
			int tlx = landmarks->fcen_x - landmarks->fwidth / 2.0 - landmarks->fwidth / 10.0; // go a little further than the box-width
			int tly = landmarks->fcen_y - landmarks->fheight / 2.0 - landmarks->fheight / 10.0;
			int w = landmarks->fwidth + landmarks->fwidth / 5.0;
			int h = landmarks->fheight + landmarks->fheight / 5.0;
			if (tlx < 0 || tlx + w >= frame.cols || tly < 0 || tly + h >= frame.rows) {
				// patch has some regions outside the image
				appLogger.debug("Had to throw away the positive patch because it goes outside the image bounds.");
				continue;
			}
			cv::Rect roi(tlx, tly, w, h);
			Mat croppedFace = frame(roi);
			cv::resize(croppedFace, croppedFace, cv::Size(32, 32)); // need to set this higher... 40x40? What about aspect ratio? At the moment we change it!
			cv::cvtColor(croppedFace, croppedFace, cv::COLOR_BGR2GRAY);

			trainingFrames.emplace_back(croppedFace);
			labels.emplace_back(frameScore);
		
			for (int k = 0; k < numNegativePairsPerFrame; ++k) {
				double recognitionScore;
				double frameScore = 1.0 - recognitionScore; // This is our label. It's a negative pair, so a facerec-score close to 0 is good, meaning 1-frScore is the frameScore
				// This is problematic here because we return a score of 0 for frames that we couldn't enrol - they would get 0 error here, which is bad.
			}

			// From this pair:
			// resulting score difference to class label = label, facebox = input, resize it
			//trainingFrames.emplace_back(frame); // store only the resized frames!
			//labels.emplace_back(1.0f); // 1.0 means good frame, 0.0 means bad frame.
		}

	}

	std::ofstream ofPascT((outputPath / "frames_data.txt").string());
	{ // use scope to ensure archive goes out of scope before stream
		boost::archive::text_oarchive oa(ofPascT);
		oa << trainingFrames;
	}
	ofPascT.close();

	std::ofstream ofPascT2((outputPath / "frames_labels.txt").string());
	{ // use scope to ensure archive goes out of scope before stream
		boost::archive::text_oarchive oa(ofPascT2);
		oa << labels;
	}
	ofPascT2.close();

	return EXIT_SUCCESS;
}
