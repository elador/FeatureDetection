/*
 * sdmEvaluation.cpp
 *
 *  Created on: 25.03.2014
 *      Author: Patrik Huber
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

#include <chrono>
#include <memory>
#include <iostream>

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

#include "superviseddescentmodel/SdmLandmarkModel.hpp"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/CameraImageSource.hpp"
#include "imageio/NamedLabeledImageSource.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/IbugLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
using namespace superviseddescentmodel;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using boost::property_tree::ptree;
using boost::filesystem::path;
using boost::lexical_cast;
using cv::Mat;
using cv::Vec2f;
using cv::Point2f;
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
	bool useFileList = false;
	bool useImgs = false;
	bool useDirectory = false;
	vector<path> inputPaths;
	path inputFilelist;
	path inputDirectory;
	vector<path> inputFilenames;
	shared_ptr<ImageSource> imageSource;
	path landmarksDir; // TODO: Make more dynamic wrt landmark format. a) What about the loading-flags (1_Per_Folder etc) we have? b) Expose those flags to cmdline? c) Make a LmSourceLoader and he knows about a LM_TYPE (each corresponds to a Parser/Loader class?)
	string landmarkType;
	path sdmModelFile;
	path faceDetectorFilename;
	bool trackingMode;
	path outputDirectory;
	vector<string> landmarksToCompare;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("input,i", po::value<vector<path>>(&inputPaths)/*->required()*/, 
				"input from one or more files, a directory, or a  .lst/.txt-file containing a list of images")
			("model,m", po::value<path>(&sdmModelFile)->required(),
				"A SDM model file to load.")
			("face-detector,f", po::value<path>(&faceDetectorFilename)->required(),
				"Path to an XML CascadeClassifier from OpenCV.")
			("landmarks,l", po::value<path>(&landmarksDir), 
				"load landmark files from the given folder")
			("landmark-type,t", po::value<string>(&landmarkType), 
				"specify the type of landmarks to load: ibug")
			("landmarks-to-compare,c", po::value<vector<string>>(&landmarksToCompare)->multitoken()->required(),
				"todo")
			("output,o", po::value<path>(&outputDirectory)->required(),
				"Output dir for imgs + 1 lm-err norm. by IED file.")
		;

		po::positional_options_description p;
		p.add("input", -1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << "Usage: sdmEvaluation [options]\n";
			cout << desc;
			return EXIT_SUCCESS;
		}

	} catch(std::exception& e) {
		cout << e.what() << endl;
		return EXIT_FAILURE;
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
	
	Loggers->getLogger("superviseddescentmodel").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("sdmEvaluation").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("sdmEvaluation");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	if (inputPaths.size() > 1) {
		// We assume the user has given several, valid images
		useImgs = true;
		inputFilenames = inputPaths;
	} else if (inputPaths.size() == 1) {
		// We assume the user has given either an image, directory, or a .lst-file
		if (inputPaths[0].extension().string() == ".lst" || inputPaths[0].extension().string() == ".txt") { // check for .lst or .txt first
			useFileList = true;
			inputFilelist = inputPaths.front();
		} else if (boost::filesystem::is_directory(inputPaths[0])) { // check if it's a directory
			useDirectory = true;
			inputDirectory = inputPaths.front();
		} else { // it must be an image
			useImgs = true;
			inputFilenames = inputPaths;
		}
	} else {
		appLogger.error("Please either specify one or several files, a directory, or a .lst-file containing a list of images to run the program!");
		return EXIT_FAILURE;
	}

	if (useFileList==true) {
		appLogger.info("Using file-list as input: " + inputFilelist.string());
		shared_ptr<ImageSource> fileListImgSrc; // TODO VS2013 change to unique_ptr, rest below also
		try {
			fileListImgSrc = make_shared<FileListImageSource>(inputFilelist.string());
		} catch(const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
		imageSource = fileListImgSrc;
	}
	if (useImgs==true) {
		//imageSource = make_shared<FileImageSource>(inputFilenames);
		//imageSource = make_shared<RepeatingFileImageSource>("C:\\Users\\Patrik\\GitHub\\data\\firstrun\\ws_8.png");
		appLogger.info("Using input images: ");
		vector<string> inputFilenamesStrings;	// Hack until we use vector<path> (?)
		for (const auto& fn : inputFilenames) {
			appLogger.info(fn.string());
			inputFilenamesStrings.push_back(fn.string());
		}
		shared_ptr<ImageSource> fileImgSrc;
		try {
			fileImgSrc = make_shared<FileImageSource>(inputFilenamesStrings);
		} catch(const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
		imageSource = fileImgSrc;
	}
	if (useDirectory==true) {
		appLogger.info("Using input images from directory: " + inputDirectory.string());
		try {
			imageSource = make_shared<DirectoryImageSource>(inputDirectory.string());
		} catch(const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
	}

	// Load the ground truth
	// Either a) use if/else for imageSource or labeledImageSource, or b) use an EmptyLandmarkSoure
	shared_ptr<LabeledImageSource> labeledImageSource;
	shared_ptr<NamedLandmarkSource> landmarkSource;
	
	vector<path> groundtruthDirs; groundtruthDirs.push_back(landmarksDir); // Todo: Make cmdline use a vector<path>
	shared_ptr<LandmarkFormatParser> landmarkFormatParser;
	if(boost::iequals(landmarkType, "lst")) {
		//landmarkFormatParser = make_shared<LstLandmarkFormatParser>();
		//landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, string(), GatherMethod::SEPARATE_FILES, groundtruthDirs), landmarkFormatParser);
	} else if(boost::iequals(landmarkType, "ibug")) {
		landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
		landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, ".pts", GatherMethod::ONE_FILE_PER_IMAGE_SAME_DIR, groundtruthDirs), landmarkFormatParser);
	} else {
		cout << "Error: Invalid ground truth type." << endl;
		return EXIT_FAILURE;
	}
	
	labeledImageSource = make_shared<NamedLabeledImageSource>(imageSource, landmarkSource);
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;

	vector<imageio::ModelLandmark> landmarks;

	SdmLandmarkModel lmModel = SdmLandmarkModel::load(sdmModelFile);
	SdmLandmarkModelFitting modelFitter(lmModel);

	// faceDetectorFilename: e.g. opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(faceDetectorFilename.string()))
	{
		cout << "Error loading face detection model." << endl;
		return EXIT_FAILURE;
	}
	
	// TODO: If landmarksToCompare is empty, fill it to use all

	std::ofstream resultsFile((outputDirectory / "results.txt").string());

	while(labeledImageSource->next()) {
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + labeledImageSource->getName().string());
		img = labeledImageSource->getImage();
		Mat landmarksImage = img.clone();

		LandmarkCollection groundtruth = labeledImageSource->getLandmarks();
		vector<shared_ptr<Landmark>> lmv = groundtruth.getLandmarks();
		for (const auto& l : lmv) {
			cv::circle(landmarksImage, l->getPoint2D(), 3, Scalar(255.0f, 0.0f, 0.0f));
		}

		// iBug 68 points. No eye-centers. Calculate them:
		cv::Point2f reye_c = (groundtruth.getLandmark("37")->getPosition2D() + groundtruth.getLandmark("40")->getPosition2D()) / 2.0f;
		cv::Point2f leye_c = (groundtruth.getLandmark("43")->getPosition2D() + groundtruth.getLandmark("46")->getPosition2D()) / 2.0f;
		cv::circle(landmarksImage, reye_c, 3, Scalar(255.0f, 0.0f, 127.0f));
		cv::circle(landmarksImage, leye_c, 3, Scalar(255.0f, 0.0f, 127.0f));
		cv::Scalar interEyeDistance = cv::norm(Vec2f(reye_c), Vec2f(leye_c), cv::NORM_L2);

		Mat imgGray;
		cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
		vector<cv::Rect> faces;
		float score, notFace = 0.5;
		
		// face detection
		faceCascade.detectMultiScale(img, faces, 1.2, 2, 0, cv::Size(50, 50));
		if (faces.empty()) {
			// no face found. Skip this image.
			continue;
		}
		for (const auto& f : faces) {
			cv::rectangle(landmarksImage, f, cv::Scalar(0.0f, 0.0f, 255.0f));
		}

		// Skip any face with w or h < 20
		vector<cv::Rect> definiteFaces;
		for (const auto& f : faces) {
			if (f.width >= 20 && f.height >= 20) {
				definiteFaces.push_back(f);
			}
		}
		faces = definiteFaces;

		// calculate the center of the bounding boxes
		vector<Point2f> faceboxCenters;
		for (const auto& f : faces) {
			faceboxCenters.push_back(Point2f(f.x + f.width/2.0f, f.y + f.height/2.0f));
		}

		// calculate the center of the ground-truth landmarks
		Mat gtLmsRowX(1, lmv.size(), CV_32FC1);
		Mat gtLmsRowY(1, lmv.size(), CV_32FC1);
		int idx = 0;
		for (const auto& l : lmv) {
			gtLmsRowX.at<float>(idx) = l->getX();
			gtLmsRowY.at<float>(idx) = l->getY();
			++idx;
		}
		double minWidth, maxWidth, minHeight, maxHeight;
		cv::minMaxIdx(gtLmsRowX, &minWidth, &maxWidth);
		cv::minMaxIdx(gtLmsRowY, &minHeight, &maxHeight);
		cv::Point2f groundtruthCenter(cv::mean(gtLmsRowX)[0], cv::mean(gtLmsRowY)[0]); // we could subtract some offset here from 'y' (w.r.t. the IED) because the center of mass of the landmarks is below the face-box center of V&J.
		
		vector<float> gtToFbDistances;
		for (const auto& f : faceboxCenters) {
			cv::Scalar distance = cv::norm(Vec2f(f), Vec2f(groundtruthCenter), cv::NORM_L2);
			gtToFbDistances.push_back(distance[0]);
		}

		const auto minDistIt = std::min_element(std::begin(gtToFbDistances), std::end(gtToFbDistances));
		const auto minDistIdx = std::distance(std::begin(gtToFbDistances), minDistIt);
		
		if (gtToFbDistances[minDistIdx] > ((maxWidth-minWidth) + (maxHeight-minHeight)) / 2.0f || faces[minDistIdx].width*1.2f < (maxWidth-minWidth)) {
			// the chosen facebox is smaller than the max-width of the ground-truth landmarks (slightly adjusted because the V&J fb seems rather small)
			// or
			// the center of the chosen facebox is further away than the avg(width+height) of the gt (i.e. the point is outside the bbox enclosing the gt-lms)
			// ==> skip the image
			continue;
		}
		
		Mat modelShape = lmModel.getMeanShape();
		modelShape = modelFitter.alignRigid(modelShape, faces[minDistIdx]);
	
		// print the mean initialization
		for (int i = 0; i < lmModel.getNumLandmarks(); ++i) {
			cv::circle(landmarksImage, Point2f(modelShape.at<float>(i, 0), modelShape.at<float>(i + lmModel.getNumLandmarks(), 0)), 3, Scalar(255.0f, 0.0f, 255.0f));
		}
		modelShape = modelFitter.optimize(modelShape, imgGray);
		for (int i = 0; i < lmModel.getNumLandmarks(); ++i) {
			cv::circle(landmarksImage, Point2f(modelShape.at<float>(i, 0), modelShape.at<float>(i + lmModel.getNumLandmarks(), 0)), 3, Scalar(0.0f, 255.0f, 0.0f));
		}

		imwrite((outputDirectory / labeledImageSource->getName().filename()).string(), landmarksImage);

		resultsFile << "# " << labeledImageSource->getName() << std::endl;
		for (const auto& lmId : landmarksToCompare) {

			shared_ptr<Landmark> gtlm = groundtruth.getLandmark(lmId); // Todo: Handle case when LM not found
			cv::Point2f gt = gtlm->getPoint2D();
			cv::Point2f det = lmModel.getLandmarkAsPoint(lmId, modelShape);

			float dx = (gt.x - det.x);
			float dy = (gt.y - det.y);
			float diff = std::sqrt(dx*dx + dy*dy);
			diff = diff / interEyeDistance[0]; // normalize by the IED

			resultsFile << diff << " # " << lmId << std::endl;
		}

		
		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");
		
	}

	resultsFile.close();

	return 0;
}
