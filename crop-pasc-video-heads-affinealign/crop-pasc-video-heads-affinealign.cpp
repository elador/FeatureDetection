/*
 * crop-pasc-video-heads-affinealign.cpp
 *
 *  Created on: 28.10.2014
 *      Author: Patrik Huber
 *
 * Example:
 * crop-pasc-video-heads-affinealign ...
 *   
 */

#include <memory>
#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
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
#include "boost/range.hpp"
#include "boost/range/algorithm.hpp"
#include "boost/range/adaptor/filtered.hpp"

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

Mat equaliseIntensity(const Mat& inputImage)
{
	if (inputImage.channels() >= 3) {
		Mat ycrcb;

		cvtColor(inputImage, ycrcb, CV_BGR2YCrCb);

		vector<Mat> channels;
		split(ycrcb, channels);

		equalizeHist(channels[0], channels[0]);

		Mat result;
		merge(channels, ycrcb);

		cvtColor(ycrcb, result, CV_YCrCb2BGR);

		return result;
	}
	return Mat();
}

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
			cout << "Usage: crop-pasc-video-heads-affinealign [options]" << endl;
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
		//std::copy(fs::directory_iterator(inputDirectory), fs::directory_iterator(), std::back_inserter(files));
		/*std::copy(fs::directory_iterator(inputDirectory), fs::directory_iterator(), [&files](path p){ if (p.extension() == ".png") {
			files.emplace_back(p);
		}
		});*/
		//std::vector<fs::directory_entry> pngFiles;
		//boost::copy(boost::make_iterator_range(fs::directory_iterator(inputDirectory), fs::directory_iterator()) | boost::adaptors::filtered([](const path& p) {return (p.extension() == ".png"); }), std::back_inserter(pngFiles));

		fs::directory_iterator end_itr;
		for (boost::filesystem::directory_iterator i(inputDirectory); i != end_itr; ++i)
		{
			if (fs::is_regular_file(i->status()) && i->path().extension() == ".png")
				files.emplace_back(i->path());
		}

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
			string logMessage("Frame has no PittPatt detections in the metadata file. This should only happen in a few videos where we don't have metadata even for a single frame (which means the frameselection will just output the first or a random frame. Skipping this image!");
			appLogger.warn(logMessage);
			//throw std::runtime_error(logMessage);
			continue;
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

		if (!landmarks->re_x || !landmarks->re_y || !landmarks->le_x || !landmarks->le_x) {
			// Hmm no PP eyes. How are we going to do the affine alignment?
			string logMessage("Hmm no PP eyes. How are we going to do the affine alignment? Skipping the image for now.");
			appLogger.error(logMessage);
			//throw std::runtime_error(logMessage);
			continue;
		}
		//cv::circle(frame, cv::Point(landmarks->re_x.get(), landmarks->re_y.get()), 2, { 255.0, 0.0, 0.0 });
		//cv::circle(frame, cv::Point(landmarks->le_x.get(), landmarks->le_y.get()), 2, { 255.0, 0.0, 0.0 });
		
		cv::Vec2f re(landmarks->re_x.get(), landmarks->re_y.get());
		cv::Vec2f le(landmarks->le_x.get(), landmarks->le_y.get());

		// Angle calc:
		cv::Vec2f reToLeLandmarksLine(le[0] - re[0], le[1] - re[1]);
		float angle = std::atan2(reToLeLandmarksLine[1], reToLeLandmarksLine[0]);
		float angleDegrees = angle * (180.0 / 3.141592654);
		// IED:
		float ied = cv::norm(reToLeLandmarksLine, cv::NORM_L2);

		// Rotate it:
		cv::Vec2f centerOfRotation = (re + le) / 2; // between the eyes
		Mat rotationMatrix = cv::getRotationMatrix2D(centerOfRotation, angleDegrees, 1.0f);
		cv::Mat rotatedFrame;
		cv::warpAffine(frame, rotatedFrame, rotationMatrix, rotatedFrame.size(), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT);

		// Crop, place eyes in "middle" horizontal, and at 1/3 vertical
		float widthFactor = 1.1f; // total 2.2
		float heightFactor = 0.8f; // total 2.4
		cv::Rect roi(centerOfRotation[0] - widthFactor * ied, centerOfRotation[1] - heightFactor * ied, 2 * widthFactor * ied, (heightFactor + 2 * heightFactor) * ied);
		Mat croppedFace = rotatedFrame(roi);

		// Normalise:
		croppedFace = equaliseIntensity(croppedFace);
		
		/*
		Mat mask(0, 0, IPL_DEPTH_8U, 1);
		Mat gr;
		Mat gf; // out
		cv::cvtColor(croppedFace, gr, cv::COLOR_BGR2GRAY);
		// gamma stuff:
		Mat temp;
		cvConvertScale(gr, temp, 1.0 / 255, 0);
		cvPow(temp, temp, 0.2);
		cvConvertScale(temp, gf, 255, 0);

		cvSmooth(gf, b1, CV_GAUSSIAN, 1);
		cvSmooth(gf, b2, CV_GAUSSIAN, 23);
		cvSub(b1, b2, b2, mask);
		cvConvertScale(b2, gr, 127, 127);
		cvEqualizeHist(gr, gr); // ==> use CvNormalize
		//cvThreshold(gr,tr,255,0,CV_THRESH_TRUNC);
		*/

		path croppedImageFilename = outputFolder / file.filename();
		cv::imwrite(croppedImageFilename.string(), croppedFace);
	}
	appLogger.info("Finished cropping all files in the directory.");

	return EXIT_SUCCESS;
}
