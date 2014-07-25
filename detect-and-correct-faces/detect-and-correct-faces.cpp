/*
 * detect-and-correct-faces.cpp
 *
 *  Created on: 04.07.2014
 *      Author: Patrik Huber
 *
 *  Example command-line arguments to run:
 * -v -i C:\\Users\\Patrik\\Documents\\Github\\data\\iBug_lfpw\\testset\\ -g "C:\Users\Patrik\Documents\GitHub\data\iBug_lfpw\trainset" -t ibug -f "C:\opencv\opencv_2.4.8_prebuilt\sources\data\haarcascades\haarcascade_frontalface_alt2.xml" -o C:\\Users\\Patrik\\Documents\\GitHub\\data\\labels\\ibug-lfpw\\automatic-OpenCV-VJ-validonly
 */

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

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

#include <chrono>
#include <ctime>
#include <memory>
#include <iostream>
#include <random>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/lexical_cast.hpp"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/NamedLabeledImageSource.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/IbugLandmarkFormatParser.hpp"
#include "imageio/PascStillEyesLandmarkFormatParser.hpp"
#include "imageio/RectLandmark.hpp"
#include "imageio/RectLandmarkSink.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using std::map;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using boost::property_tree::ptree;
using boost::filesystem::path;
using boost::lexical_cast;
using cv::Mat;
using cv::Rect;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;

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
	vector<path> inputPaths;
	path groundtruthPath, faceDetectorFilename, outputDirectory;
	string groundtruthType;
	bool doOutputImages;

	bool useFileList = false;
	bool useImgs = false;
	bool useDirectory = false;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("input,i", po::value<vector<path>>(&inputPaths)->required(),
				"input from one or more files, a directory, or a  .lst/.txt-file containing a list of images")
			("groundtruth,g", po::value<path>(&groundtruthPath)->required(),
				"groundtruth landmarks to validate found faces")
			("groundtruth-type,t", po::value<string>(&groundtruthType)->required(),
				"specify the type of landmarks to load: ibug, PaSC-still-PittPatt-eyes")
			("face-detector,f", po::value<path>(&faceDetectorFilename)->required(),
				"Path to an XML CascadeClassifier from OpenCV.")
			("output,o", po::value<path>(&outputDirectory)->default_value("."),
				"output folder to write the detected face boxes to")
			("output-images,p", po::value<bool>(&doOutputImages)->default_value(false),
				"true or false, write the detected face and the ground-truth landmarks alongside the face box output")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: detect-and-correct-faces [options]" << endl;
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
		cout << "Error: Invalid log level." << endl;
		return EXIT_SUCCESS;
	}
	
	Loggers->getLogger("detect-and-correct-faces").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("detect-and-correct-faces");
	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Prepare the input image(s):
	if (inputPaths.size() > 1) {
		// We assume the user has given several, valid images
		useImgs = true;
	}
	else if (inputPaths.size() == 1) {
		// We assume the user has given either an image, directory, or a .lst-file
		if (inputPaths[0].extension().string() == ".lst" || inputPaths[0].extension().string() == ".txt") { // check for .lst or .txt first
			useFileList = true;
		}
		else if (boost::filesystem::is_directory(inputPaths[0])) { // check if it's a directory
			useDirectory = true;
		}
		else { // it must be an image
			useImgs = true;
		}
	}
	else {
		appLogger.error("Please either specify one or several files, a directory, or a .lst-file containing a list of images to run the program!");
		return EXIT_FAILURE;
	}
	
	shared_ptr<ImageSource> imageSource;
	if (useFileList == true) {
		appLogger.info("Using file-list as input: " + inputPaths.front().string());
		shared_ptr<ImageSource> fileListImgSrc; // TODO VS2013 change to unique_ptr, rest below also
		try {
			fileListImgSrc = make_shared<FileListImageSource>(inputPaths.front().string());
		}
		catch (const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
		imageSource = fileListImgSrc;
	}
	if (useImgs == true) {
		appLogger.info("Using input images: ");
		vector<string> inputFilenamesStrings;	// Hack until we use vector<path> (?)
		for (const auto& fn : inputPaths) {
			appLogger.info(fn.string());
			inputFilenamesStrings.push_back(fn.string());
		}
		shared_ptr<ImageSource> fileImgSrc;
		try {
			fileImgSrc = make_shared<FileImageSource>(inputFilenamesStrings);
		}
		catch (const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
		imageSource = fileImgSrc;
	}
	if (useDirectory == true) {
		appLogger.info("Using input images from directory: " + inputPaths.front().string());
		try {
			imageSource = make_shared<DirectoryImageSource>(inputPaths.front().string());
		}
		catch (const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
	}

	// Prepare the ground truth landmarks:
	shared_ptr<NamedLandmarkSource> groundtruthSource;
	vector<path> groundtruthDirs{ groundtruthPath };
	shared_ptr<LandmarkFormatParser> landmarkFormatParser;
	if (boost::iequals(groundtruthType, "ibug")) {
		landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
		groundtruthSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(nullptr, ".pts", GatherMethod::SEPARATE_FOLDERS, groundtruthDirs), landmarkFormatParser);
	}
	else if (boost::iequals(groundtruthType, "PaSC-still-PittPatt-eyes")) {
		landmarkFormatParser = make_shared<PascStillEyesLandmarkFormatParser>();
		groundtruthSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(nullptr, ".csv", GatherMethod::SEPARATE_FILES, groundtruthDirs), landmarkFormatParser);
	}
	else {
		appLogger.error("Error: Invalid ground-truth landmarks type.");
		return EXIT_FAILURE;
	}

	// Face detector:
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(faceDetectorFilename.string()))
	{
		appLogger.error("Error loading the face detection model.");
		return EXIT_FAILURE;
	}

	// Output directory and sink:
	if (!boost::filesystem::exists(outputDirectory)) {
		boost::filesystem::create_directory(outputDirectory);
	}
	RectLandmarkSink landmarkSink(outputDirectory);

	std::chrono::time_point<std::chrono::system_clock> start, end;
	while (imageSource->next()) {
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + imageSource->getName().string());
		Mat img = imageSource->getImage();
		Mat landmarksImage = img.clone();

		Mat imgGray;
		cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
		vector<cv::Rect> faces;
		float score, notFace = 0.5;

		// face detection
		//faceCascade.detectMultiScale(img, faces, 1.2, 2, 0, cv::Size(50, 50));
		faceCascade.detectMultiScale(img, faces);
		if (faces.empty()) {
			// no face found, output nothing
			continue;
		}
		// draw the best face candidate
		cv::rectangle(landmarksImage, faces[0], cv::Scalar(0.0f, 0.0f, 255.0f));

		LandmarkCollection groundtruthLandmarks = groundtruthSource->get(imageSource->getName());

		// Start duplicate --- the following is 100% the same (except the params?) as in landmarkEvaluation.cpp. Move that to a function (libLandmarkDetection helpers?)
		// See if we discard the detection result:
		// Ideally, we'd use the detected V&J face box here and skip any face with w or h < 20.
		// If the actual detection and evaluation were in one program, this would also allow us to select the correct box for landmark detection, if V&J finds several. We would then select the one where the box-centers are closest.
		// But as we don't have it (without further modifications), we use the face box approximated from the landmarks.
		cv::Rect groundtruthFacebox = getBoundingBox(groundtruthLandmarks);
		
		cv::rectangle(landmarksImage, groundtruthFacebox, cv::Scalar(255.0f, 0.0f, 0.0f)); ////
		for (auto&& lm : groundtruthLandmarks.getLandmarks()) {
			lm->draw(landmarksImage);
		}
		
		cv::Rect detectedFacebox = faces[0];
		cv::Vec2f groundtruthCenter(groundtruthFacebox.x + groundtruthFacebox.width / 2.0f, groundtruthFacebox.y + groundtruthFacebox.height / 2.0f);
		cv::Vec2f detectedCenter(detectedFacebox.x + detectedFacebox.width / 2.0f, detectedFacebox.y + detectedFacebox.height / 2.0f);

		cv::Scalar distance = cv::norm(groundtruthCenter, detectedCenter, cv::NORM_L2);

		// Write out the output image for every input image, also the ones we reject. Except of no face is found.
		if (doOutputImages) {
			cv::imwrite((outputDirectory / imageSource->getName().filename()).string(), landmarksImage);
		}

		if (detectedFacebox.width < 25 || detectedFacebox.height < 25) { // Todo: Those params could go into the config.
			continue;
		}

		if (distance[0] > (groundtruthFacebox.width + groundtruthFacebox.height) / 4.0f || detectedFacebox.width * 1.5f < groundtruthFacebox.width) {
			// the center of the chosen facebox is further away than half the avg(width+height) of the gt (i.e. the detected center-point is outside the bbox enclosing the gt-lms)
			// or
			// the chosen facebox is smaller than the max-width of the ground-truth landmarks (slightly adjusted because the V&J fb seems rather small (really?))
			// ==> skip the image
			continue;
		}
		// End duplicate

		// Our facebox is valid, write out the face box landmark to a file:
		LandmarkCollection facebox;
		facebox.insert(make_shared<imageio::RectLandmark>("face", faces[0]));
		landmarkSink.write(facebox, imageSource->getName());

		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.");
	}

	return 0;
}
