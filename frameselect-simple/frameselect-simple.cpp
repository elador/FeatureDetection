/*
 * frameselect-simple.cpp
 *
 *  Created on: 21.10.2014
 *      Author: Patrik Huber
 *
 * Example:
 * frameselect-simple -v -s "C:\Users\Patrik\Documents\GitHub\data\PaSC\Protocol\PaSC_20130611\PaSC\metadata\sigsets\pasc_video_handheld.xml" -d "Z:\datasets\multiview02\PaSC\video" -l "C:\Users\Patrik\Documents\GitHub\data\PaSC\pasc_video_pittpatt_detections.txt" -o out
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
#include <fstream>
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
#include "boost/archive/text_iarchive.hpp"

#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"
#include "facerecognition/ThreadPool.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
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
//using std::pair;
using std::tuple;
using std::future;

float sharpnessScoreCanny(cv::Mat frame)
{
	// Normalise? (Histo, ZM/UV?) Contrast-normalisation?
	Mat cannyEdges;
	cv::Canny(frame, cannyEdges, 225.0, 175.0); // threshold1, threshold2
	int numEdgePixels = cv::countNonZero(cannyEdges); // throws if 0 nonZero? Check first?

	float sharpness = numEdgePixels * 1000.0f / (cannyEdges.rows * cannyEdges.cols);

	// We'll normalise the sharpness later, per video
	return sharpness;
};

// Should also try this one: http://stackoverflow.com/a/7767755/1345959
// Focus-measurement algorithms, from http://stackoverflow.com/a/7768918/1345959:

// OpenCV port of 'LAPM' algorithm (Nayar89)
double modifiedLaplacian(const cv::Mat& src)
{
	cv::Mat M = (cv::Mat_<double>(3, 1) << -1, 2, -1);
	cv::Mat G = cv::getGaussianKernel(3, -1, CV_64F);

	cv::Mat Lx;
	cv::sepFilter2D(src, Lx, CV_64F, M, G);

	cv::Mat Ly;
	cv::sepFilter2D(src, Ly, CV_64F, G, M);

	cv::Mat FM = cv::abs(Lx) + cv::abs(Ly);

	double focusMeasure = cv::mean(FM).val[0];
	return focusMeasure;
}

// OpenCV port of 'LAPV' algorithm (Pech2000)
double varianceOfLaplacian(const cv::Mat& src)
{
	cv::Mat lap;
	cv::Laplacian(src, lap, CV_64F);

	cv::Scalar mu, sigma;
	cv::meanStdDev(lap, mu, sigma);

	double focusMeasure = sigma.val[0] * sigma.val[0];
	return focusMeasure;
}

// OpenCV port of 'TENG' algorithm (Krotkov86)
double tenengrad(const cv::Mat& src, int ksize)
{
	cv::Mat Gx, Gy;
	cv::Sobel(src, Gx, CV_64F, 1, 0, ksize);
	cv::Sobel(src, Gy, CV_64F, 0, 1, ksize);

	cv::Mat FM = Gx.mul(Gx) + Gy.mul(Gy);

	double focusMeasure = cv::mean(FM).val[0];
	return focusMeasure;
}

// OpenCV port of 'GLVN' algorithm (Santos97)
double normalizedGraylevelVariance(const cv::Mat& src)
{
	cv::Scalar mu, sigma;
	cv::meanStdDev(src, mu, sigma);

	double focusMeasure = (sigma.val[0] * sigma.val[0]) / mu.val[0];
	return focusMeasure;
}

// Could all be renamed to "transform/fitLinear" or something
// Note: Segfaults if the given vector is empty (iterator not dereferentiable)
vector<float> getVideoNormalizedHeadBoxScores(vector<float> headBoxSizes)
{
	auto result = std::minmax_element(begin(headBoxSizes), end(headBoxSizes));
	auto min = *result.first;
	auto max = *result.second;

	float m = 1.0f / (max - min);
	float b = -m * min;

	std::transform(begin(headBoxSizes), end(headBoxSizes), begin(headBoxSizes), [m, b](float x) {return m * x + b; });

	return headBoxSizes;
}

vector<float> getVideoNormalizedInterEyeDistanceScores(vector<float> ieds)
{
	auto result = std::minmax_element(begin(ieds), end(ieds));
	auto min = *result.first;
	auto max = *result.second;

	float m = 1.0f / (max - min);
	float b = -m * min;

	std::transform(begin(ieds), end(ieds), begin(ieds), [m, b](float x) {return m * x + b; });

	return ieds;
}

vector<float> getVideoNormalizedYawPoseScores(vector<float> yaws)
{
	// We work on the absolute angle values
	std::transform(begin(yaws), end(yaws), begin(yaws), [](float x) {return std::abs(x); });

	// Actually, for the yaw we want an absolute scale and not normalise per video!
	float m = -1.0f / 30.0f; // (30 = 40 - 10) (at 40 = 0, at 10 = 1)
	float b = -40.0f * m; // at 40 = 0
	for (auto& e : yaws) {
		if (e <= 10.0f) {
			e = 1.0f;
		}
		else {
			e = m * e + b;
		}
	}
	return yaws;
}

vector<float> getVideoNormalizedCannySharpnessScores(vector<float> sharpnesses)
{
	auto result = std::minmax_element(begin(sharpnesses), end(sharpnesses));
	auto min = *result.first;
	auto max = *result.second;

	float m = 1.0f / (max - min);
	float b = -m * min;

	std::transform(begin(sharpnesses), end(sharpnesses), begin(sharpnesses), [m, b](float x) {return m * x + b; });

	return sharpnesses;
}

// Might rename to "assessQualitySimple(video...)" or assessFrames(...)
// Assesses all or part of the frames in the video (e.g. not those without metadata)
// and returns a score for each.
std::vector<std::tuple<cv::Mat, path, float>> selectFrameSimple(path inputDirectoryVideos, const facerecognition::FaceRecord& video, const vector<facerecognition::PascVideoDetection>& pascVideoDetections)
{
	auto logger = Loggers->getLogger("frameselect-simple");

	auto frames = facerecognition::utils::getFrames(inputDirectoryVideos / video.dataPath);
	vector<float> headBoxSizes;
	vector<float> ieds;
	vector<float> yaws;
	vector<float> sharpnesses;
	vector<float> laplModif;
	vector<float> laplVari;
	vector<int> frameIds;
	for (int frameNum = 0; frameNum < frames.size(); ++frameNum) {
		string frameName = facerecognition::getPascFrameName(video.dataPath, frameNum + 1);
		logger.debug("Processing frame " + frameName);
		auto landmarks = std::find_if(begin(pascVideoDetections), end(pascVideoDetections), [frameName](const facerecognition::PascVideoDetection& d) { return (d.frame_id == frameName); });
		// If we were to run it on the training-videos, we'd have to test if we got the eyes and facebox?
		// The face box is always given. Only the eyes are missing sometimes.
		// For the test-videos, the whole line is just missing for frames without annotations.
		if (landmarks == std::end(pascVideoDetections)) {
			logger.debug("Frame has no PittPatt detections in the metadata file.");
			continue;
		}
		int tlx = landmarks->fcen_x - landmarks->fwidth / 2.0;
		int tly = landmarks->fcen_y - landmarks->fheight / 2.0;
		int w = landmarks->fwidth;
		int h = landmarks->fheight;
		if (tlx < 0 || tlx + w >= frames[frameNum].cols || tly < 0 || tly + h >= frames[frameNum].rows) {
			// patch has some regions outside the image
			logger.debug("Throwing away patch because it goes outside the image bounds.");
			continue;
		}
		cv::Rect roi(tlx, tly, w, h);
		Mat croppedFace = frames[frameNum](roi);

		headBoxSizes.emplace_back((landmarks->fwidth + landmarks->fheight) / 2.0f);
		ieds.emplace_back(cv::norm(cv::Vec2f(*landmarks->le_x, *landmarks->le_y), cv::Vec2f(*landmarks->re_x, *landmarks->re_y), cv::NORM_L2));
		yaws.emplace_back(landmarks->fpose_y);
		sharpnesses.emplace_back(sharpnessScoreCanny(croppedFace));
		frameIds.emplace_back(frameNum);

		laplModif.emplace_back(modifiedLaplacian(croppedFace));
		laplVari.emplace_back(varianceOfLaplacian(croppedFace));
		/*{
			cv::imwrite("out/out_" + std::to_string(frameIds.size() - 1) + "_" + std::to_string(frameNum) + ".png", croppedFace);
			Mat cannyEdges;
			cv::Canny(croppedFace, cannyEdges, 225.0, 175.0); // threshold1, threshold2
			int numEdgePixels = cv::countNonZero(cannyEdges); // throws if 0 nonZero? Check first?
			float sharpness = numEdgePixels * 1000.0f / (cannyEdges.rows * cannyEdges.cols);
			cv::imwrite("out/out_" + std::to_string(frameIds.size() - 1) + "_" + std::to_string(frameNum) + "_" + std::to_string(sharpness) + ".png", cannyEdges);
			}*/
	}

	if (headBoxSizes.size() == 0) {
		// We don't have ANY frame with PaSC landmarks (or we threw it away because it's out of image borders). Happened for 1 test video so far.
		logger.warn("No metadata available for this video, or threw all available frames away.");
		int idOfBestFrame = 0;
		cv::Mat bestFrame = frames[idOfBestFrame];

		path bestFrameName = video.dataPath.stem();
		std::ostringstream ss;
		ss << std::setw(3) << std::setfill('0') << idOfBestFrame + 1;
		bestFrameName.replace_extension(ss.str() + ".png");
		return vector<std::tuple<Mat, path, float>>{ std::make_tuple(bestFrame, bestFrameName, 0.0f) };
	}

	cv::Mat headBoxScores(getVideoNormalizedHeadBoxScores(headBoxSizes), true); // function returns a temporary, so we need to copy the data
	cv::Mat interEyeDistanceScores(getVideoNormalizedInterEyeDistanceScores(ieds), true);
	cv::Mat yawPoseScores(getVideoNormalizedYawPoseScores(yaws), true);
	cv::Mat cannySharpnessScores(getVideoNormalizedCannySharpnessScores(sharpnesses), true);
	cv::Mat modifiedLaplacianInFocusScores(getVideoNormalizedCannySharpnessScores(laplModif), true);
	cv::Mat varianceOfLaplacianInFocusScores(getVideoNormalizedCannySharpnessScores(laplVari), true);
	
	// Weights:
	//  0.2 head box size
	//  0.1 IED
	//  0.4 yaw pose
	//  0.1 canny edges blur measurement
	//  0.1 modified laplace
	//  0.1 laplace variance
	cv::Mat scores = 0.2f * headBoxScores + 0.1f * interEyeDistanceScores + 0.4f * yawPoseScores + 0.1f * cannySharpnessScores + 0.1f * modifiedLaplacianInFocusScores + 0.1f * varianceOfLaplacianInFocusScores;

	/*
	double minScore, maxScore;
	int minIdx[2], maxIdx[2];
	cv::minMaxIdx(scores, &minScore, &maxScore, &minIdx[0], &maxIdx[0]); // can use NULL if argument not needed
	int idOfBestFrame = frameIds[maxIdx[0]]; // scores is a single column so it will be M x 1, i.e. we need maxIdx[0]
	cv::Mat bestFrame = frames[idOfBestFrame];
	//int idOfWorstFrame = frameIds[minIdx];
	//cv::Mat worstFrame = frames[idOfWorstFrame];
	*/

	vector<std::tuple<Mat, path, float>> assessedFrames;
	for (int i = 0; i < scores.rows; ++i) {
		path bestFrameName = video.dataPath.stem();
		std::ostringstream ss;
		ss << std::setw(3) << std::setfill('0') << frameIds[i] + 1;
		bestFrameName.replace_extension(ss.str() + ".png");
		assessedFrames.emplace_back(std::make_tuple(frames[frameIds[i]], bestFrameName, scores.at<float>(i)));
	}
	return assessedFrames;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path sigsetPath, inputDirectoryVideos, landmarksPath;
	int numFramesPerVideo;
	int numThreads;
	path outputPath;


	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("sigset,s", po::value<path>(&sigsetPath)->required(),
				"PaSC video sigset")
			("data,d", po::value<path>(&inputDirectoryVideos)->required(),
				"path to the videos")
			("landmarks,l", po::value<path>(&landmarksPath)->required(),
				"PaSC landmarks for the videos in boost::serialization text format")
			("num-frames,n", po::value<int>(&numFramesPerVideo)->default_value(1),
				"number of frames per video (maximum - actual number of assessed frames might be smaller")
			("threads,t", po::value<int>(&numThreads)->default_value(2),
				"number of threads, i.e. number of video that will be processed in parallel")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder. In addition to the frames, a CSV-file 'frameselect-simple.txt' will be written to this location")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: frameselect-simple [options]" << endl;
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
	Loggers->getLogger("frameselect-simple").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("frameselect-simple");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Read the video detections metadata (eyes, face-coords):
	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(landmarksPath.string());
		boost::archive::text_iarchive ia(ifs);
		ia >> pascVideoDetections;
	} // archive and stream closed when destructors are called

	// Read the training-video xml sigset and the training-still sigset to get the subject-id metadata:
	auto videoSigset = facerecognition::utils::readPascSigset(sigsetPath, true);

	// Create the output directory if it doesn't exist yet:
	if (!fs::exists(outputPath)) {
		fs::create_directory(outputPath);
	}

	// Read all videos:
	if (!fs::exists(inputDirectoryVideos)) {
		appLogger.error("The given input files directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}

	ThreadPool threadPool(numThreads);
	vector<future<vector<std::tuple<Mat, path, float>>>> futures;

	// If we don't want to loop over all videos: (e.g. to get a quick Matlab output)
// 	std::random_device rd;
// 	auto seed = rd();
// 	std::mt19937 rndGenVideos(seed);
// 	std::uniform_real<> rndVidDistr(0.0f, 1.0f);
// 	auto randomVideo = std::bind(rndVidDistr, rndGenVideos);
	//auto videoIter = std::find_if(begin(videoSigset), end(videoSigset), [](const facerecognition::FaceRecord& d) { return (d.dataPath == "05846d694.mp4"); });
	//auto video = *videoIter; {
	for (auto& video : videoSigset) {
// 		if (randomVideo() >= 0.002) {
// 			continue;
// 		}
		appLogger.info("Starting to process " + video.dataPath.string());

		// Shouldn't be necessary, but there are 5 videos in the xml sigset that we don't have.
		// Does it happen for the test-videos? No?
		if (!fs::exists(inputDirectoryVideos / video.dataPath)) {
			appLogger.warn("Video in the sigset not found on the filesystem!");
			continue;
		}

		futures.emplace_back(threadPool.enqueue(&selectFrameSimple, inputDirectoryVideos, video, pascVideoDetections));
	}

	std::ofstream framesListFile((outputPath / "frameselect-simple.txt").string());
	// Wait for all the threads and save the result as soon as a thread is finished:
	for (auto& f : futures) {
		auto frames = f.get();

		// Sort by score, higher ones first
		std::sort(begin(frames), end(frames), [](tuple<Mat, path, float> lhs, tuple<Mat, path, float> rhs) { return std::get<2>(rhs) < std::get<2>(lhs); });
		if (frames.size() < numFramesPerVideo) {
			numFramesPerVideo = frames.size(); // we got fewer frames than desired
		}
		for (int i = 0; i < numFramesPerVideo; ++i) {
			path frameName = outputPath / std::get<1>(frames[i]); // The filename is already 1-based (PaSC format)
			cv::imwrite(frameName.string(), std::get<0>(frames[i]));

			path frameNameCsv = frameName.stem();
			frameNameCsv.replace_extension(".mp4");
			string frameNumCsv = frameName.stem().extension().string();
			frameNumCsv.erase(std::remove(frameNumCsv.begin(), frameNumCsv.end(), '.'), frameNumCsv.end());
			framesListFile << frameNameCsv.string() << "," << frameNumCsv << "," << std::get<2>(frames[i]) << endl;
		}
		//appLogger.info("Saved frames and scores of " + frameNameCsv.string() + " to the filesystem.");
	}
	
	framesListFile.close();
	appLogger.info("Finished processing all videos.");

	return EXIT_SUCCESS;
}
