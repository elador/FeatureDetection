/*
 * sdmSimpleLandmarkDetection.cpp
 *
 *  Created on: 24.03.2014
 *      Author: Patrik Huber
 *
 *  Example command-line arguments to run:
 *    sdmSimpleLandmarkDetection -v -i /home/user/image.png -m /home/user/hogModel.txt -f /opt/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml -o /home/user/
 */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"

#include "superviseddescent/SdmLandmarkModel.hpp"

#include "logging/LoggerFactory.hpp"

using namespace superviseddescent;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using std::make_shared;
using boost::filesystem::path;
using cv::Mat;
using cv::Point2f;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;


int main(int argc, char *argv[])
{
	
	string verboseLevelConsole;
	path inputFilename;
	path sdmModelFile;
	path faceDetectorFilename;
	path outputDirectory;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"Produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "Specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("input,i", po::value<path>(&inputFilename)->required(), 
				"The input image.")
			("model,m", po::value<path>(&sdmModelFile)->required(),
				"A SDM model file to load.")
			("face-detector,f", po::value<path>(&faceDetectorFilename)->required(),
				"Path to an XML CascadeClassifier from OpenCV.")
			("output,o", po::value<path>(&outputDirectory)->required(),
				"Output directory for the result image.")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << "Usage: sdmLandmarkDetectionPlain [options]\n";
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
	
	Loggers->getLogger("superviseddescent").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("sdmSimpleLandmarkDetection").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("sdmSimpleLandmarkDetection");

	appLogger.info("Verbose level for console output: " + logging::logLevelToString(logLevel));

	if (!boost::filesystem::exists(inputFilename)) {
		appLogger.error("The input image given does not exist.");
		return EXIT_FAILURE;
	}

	SdmLandmarkModel lmModel = SdmLandmarkModel::load(sdmModelFile);
	SdmLandmarkModelFitting modelFitter(lmModel);

	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(faceDetectorFilename.string()))
	{
		appLogger.error("Error loading the face detection model.");
		return EXIT_FAILURE;
	}
	
	appLogger.info("Starting to process " + inputFilename.string());
	Mat img = cv::imread(inputFilename.string());
	Mat landmarksImage = img.clone();

	Mat imgGray;
	cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
	vector<cv::Rect> faces;
	float score, notFace = 0.5;
		
	// face detection
	faceCascade.detectMultiScale(img, faces, 1.2, 2, 0, cv::Size(50, 50));
	if (faces.empty()) {
		// no face found, output the unmodified image
		appLogger.warn("No face found, could not run landmark detection.");
		return EXIT_FAILURE;
	}
	// draw the best face candidate
	cv::rectangle(landmarksImage, faces[0], cv::Scalar(0.0f, 0.0f, 255.0f));

	// fit the model
	Mat modelShape = lmModel.getMeanShape();
	modelShape = modelFitter.alignRigid(modelShape, faces[0]);
	modelShape = modelFitter.optimize(modelShape, imgGray);

	// draw the final result
	for (int i = 0; i < lmModel.getNumLandmarks(); ++i) {
		cv::circle(landmarksImage, Point2f(modelShape.at<float>(i, 0), modelShape.at<float>(i + lmModel.getNumLandmarks(), 0)), 3, Scalar(0.0f, 255.0f, 0.0f));
	}
	// save the image
	imwrite((outputDirectory / inputFilename.filename()).string(), landmarksImage);

	appLogger.info("Finished processing.");
	return 0;
}
