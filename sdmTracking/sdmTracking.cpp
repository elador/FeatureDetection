/*
 * sdmTracking.cpp
 *
 *  Created on: 11.01.2014
 *      Author: Patrik Huber
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
#include <memory>
#include <iostream>

extern "C" {
	#include "vl/hog.h"
}

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

#include "shapemodels/MorphableModel.hpp"
#include "shapemodels/OpenCVCameraEstimation.hpp"
#include "shapemodels/AffineCameraEstimation.hpp"
#include "render/Camera.hpp"
#include "render/SoftwareRenderer.hpp"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/NamedLabeledImageSource.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/IbugLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using boost::property_tree::ptree;
using boost::filesystem::path;
using boost::lexical_cast;
using cv::Mat;
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;


template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(cout, " "));
	return os;
}



/*
#include <fstream>
#include "opencv2/core/core.hpp"
#ifdef WIN32
#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/lexical_cast.hpp"

using boost::lexical_cast;
*/
class HogSdmModel
{
public:
	HogSdmModel() {
	};

	// interface
	int getNumLandmarks() const {
		return meanLandmarks.rows/2;
	};

	// only HOG models
	int getNumHogScales() const {
		return regressorData.size();
	};

	// returns a copy
	cv::Mat getMeanShape() const {
		return meanLandmarks.clone();
	};
	// returns  a header that points to the original data
	cv::Mat getRegressorData(int hogScaleLevel) {
		return regressorData[hogScaleLevel];
	}

	//std::vector<cv::Point2f> getLandmarksAsPoints(cv::Mat or vector<float> alphas or empty(=mean));
	std::vector<cv::Point2f> getLandmarksAsPoints() const {
		std::vector<cv::Point2f> landmarks;
		for (int i = 0; i < getNumLandmarks(); ++i) {
			landmarks.push_back({ meanLandmarks.at<float>(i, 0), meanLandmarks.at<float>(i+getNumLandmarks(), 0) });
		}
		return landmarks;
	};

	static HogSdmModel load(boost::filesystem::path filename) {
		HogSdmModel model;
		std::ifstream file(filename.string());
		std::string line;
		vector<string> stringContainer;
		std::getline(file, line); // skip the first line, it's the description
		std::getline(file, line); // numLandmarks 22
		boost::split(stringContainer, line, boost::is_any_of(" "));
		int numLandmarks = lexical_cast<int>(stringContainer[1]);
		// read the mean landmarks
		model.meanLandmarks = Mat(numLandmarks*2, 1, CV_32FC1);
		// First all the x-coordinates, then all the  y-coordinates.
		for (int i = 0; i < numLandmarks*2; ++i) {
			std::getline(file, line);
			model.meanLandmarks.at<float>(i, 0) = lexical_cast<float>(line);
		}
		// read the numHogScales
		std::getline(file, line); // numHogScales 5
		boost::split(stringContainer, line, boost::is_any_of(" "));
		int numHogScales = lexical_cast<int>(stringContainer[1]);
		// for every HOG scale, read a header line and then the matrix data
		for (int i = 0; i < numHogScales; ++i) {
			// read the header line
			std::getline(file, line); // scale 1 rows 3169 cols 44
			boost::split(stringContainer, line, boost::is_any_of(" "));
			int numRows = lexical_cast<int>(stringContainer[3]); // = numHogDimensions
			int numCols = lexical_cast<int>(stringContainer[5]); // = numLandmarks * 2
			Mat regressorData(numRows, numCols, CV_32FC1);
			// read numRows lines
			for (int j = 0; j < numRows; ++j) {
				std::getline(file, line); // float1 float2 float3 ... float44
				boost::split(stringContainer, line, boost::is_any_of(" "));
				for (int col = 0; col < numCols; ++col) { // stringContainer contains one more entry than numCols, but we just skip it, it's a whitespace
					regressorData.at<float>(j, col) = lexical_cast<float>(stringContainer[col]);
				}
				
			}

			model.regressorData.push_back(regressorData);
		}

		return model;
	};

private:
	cv::Mat meanLandmarks; // numLandmarks*2 x 1. First all the x-coordinates, then all the y-coordinates.
	std::vector<cv::Mat> regressorData; // Holds the training data, one cv::Mat for each Hog scale level. Every Mat is numFeatureDim x numLandmarks*2 (for x & y)

};

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
	bool useLandmarkFiles = false;
	vector<path> inputPaths;
	path inputFilelist;
	path inputDirectory;
	vector<path> inputFilenames;
	path configFilename;
	shared_ptr<ImageSource> imageSource;
	path landmarksDir; // TODO: Make more dynamic wrt landmark format. a) What about the loading-flags (1_Per_Folder etc) we have? b) Expose those flags to cmdline? c) Make a LmSourceLoader and he knows about a LM_TYPE (each corresponds to a Parser/Loader class?)
	string landmarkType;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("config,c", po::value<path>(&configFilename)->required(), 
				"path to a config (.cfg) file")
			("input,i", po::value<vector<path>>(&inputPaths)->required(), 
				"input from one or more files, a directory, or a  .lst/.txt-file containing a list of images")
			("landmarks,l", po::value<path>(&landmarksDir), 
				"load landmark files from the given folder")
			("landmark-type,t", po::value<string>(&landmarkType), 
				"specify the type of landmarks to load: ibug")
		;

		po::positional_options_description p;
		p.add("input", -1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << "Usage: fitter [options]\n";
			cout << desc;
			return EXIT_SUCCESS;
		}
		if (vm.count("landmarks")) {
			useLandmarkFiles = true;
			if (!vm.count("landmark-type")) {
				cout << "You have specified to use landmark files. Please also specify the type of the landmarks to load via --landmark-type or -t." << endl;
				return EXIT_SUCCESS;
			}
		}

	} catch(std::exception& e) {
		cout << e.what() << endl;
		return EXIT_FAILURE;
	}

	loglevel logLevel;
	if(boost::iequals(verboseLevelConsole, "PANIC")) logLevel = loglevel::PANIC;
	else if(boost::iequals(verboseLevelConsole, "ERROR")) logLevel = loglevel::ERROR;
	else if(boost::iequals(verboseLevelConsole, "WARN")) logLevel = loglevel::WARN;
	else if(boost::iequals(verboseLevelConsole, "INFO")) logLevel = loglevel::INFO;
	else if(boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = loglevel::DEBUG;
	else if(boost::iequals(verboseLevelConsole, "TRACE")) logLevel = loglevel::TRACE;
	else {
		cout << "Error: Invalid loglevel." << endl;
		return EXIT_FAILURE;
	}
	
	Loggers->getLogger("shapemodels").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("render").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("fitter").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("fitter");

	appLogger.debug("Verbose level for console output: " + logging::loglevelToString(logLevel));
	appLogger.debug("Using config: " + configFilename.string());

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
			fileListImgSrc = make_shared<FileListImageSource>(inputFilelist.string(), "C:\\Users\\Patrik\\Documents\\GitHub\\data\\fddb\\originalPics\\", ".jpg");
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
	if (useLandmarkFiles) {
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
	} else {
		landmarkSource = make_shared<EmptyLandmarkSource>();
	}
	labeledImageSource = make_shared<NamedLabeledImageSource>(imageSource, landmarkSource);
	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(configFilename.string(), pt);
	} catch(const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	/*
	shapemodels::MorphableModel morphableModel;
	try {
		morphableModel = shapemodels::MorphableModel::load(pt.get_child("morphableModel"));
	} catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	catch (const std::runtime_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}*/
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	const string windowName = "win";

	vector<imageio::ModelLandmark> landmarks;

	cv::namedWindow(windowName);

	HogSdmModel m = HogSdmModel::load("C:\\Users\\Patrik\\Documents\\GitHub\\SGD_Zhenhua_11012014\\SDM_Model_HOG_Zhenhua_11012014.txt");
	
	string faceDetectionModel("C:\\opencv\\2.4.7.2_prebuilt\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"); // sgd: "../models/haarcascade_frontalface_alt2.xml"
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(faceDetectionModel))
	{
		cout << "Error loading face detection model." << endl;
		return EXIT_FAILURE;
	}
	
	while(labeledImageSource->next()) {
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + labeledImageSource->getName().string());
		img = labeledImageSource->getImage();
		
		LandmarkCollection lms = labeledImageSource->getLandmarks();
		vector<shared_ptr<Landmark>> lmsv = lms.getLandmarks();
		landmarks.clear();
		Mat landmarksImage = img.clone(); // blue rect = the used landmarks
		for (const auto& lm : lmsv) {
			lm->draw(landmarksImage);
			landmarks.emplace_back(imageio::ModelLandmark(lm->getName(), lm->getPosition2D()));
			cv::rectangle(landmarksImage, cv::Point(cvRound(lm->getX() - 2.0f), cvRound(lm->getY() - 2.0f)), cv::Point(cvRound(lm->getX() + 2.0f), cvRound(lm->getY() + 2.0f)), cv::Scalar(255, 0, 0));
		}

		vector<cv::Rect> faces;
		float score, notFace = 0.5;
		// face detection
		//faceCascade.detectMultiScale(img, faces, 1.2, 2, 0, cv::Size(50, 50));
		faces.push_back({ 172, 199, 278, 278 });

		for (const auto& f : faces) {
			cv::rectangle(landmarksImage, f, cv::Scalar(0.0f, 0.0f, 255.0f));
		}
		Mat imgGray;
		cvtColor(img, imgGray, cv::COLOR_RGB2GRAY);

	/*	std::vector<cv::Point2f> mlms = m.getLandmarksAsPoints();
		for (const auto& l : mlms) {
			cv::circle(landmarksImage, l, 3, Scalar(0.0f, 0.0f, 255.0f));
		}*/

		Mat modelShape = m.getMeanShape();
		Mat xCoords = modelShape.rowRange(0, modelShape.rows / 2);
		Mat yCoords = modelShape.rowRange(modelShape.rows / 2, modelShape.rows);
		// scale the model:
		double minX, maxX, minY, maxY;
		cv::minMaxLoc(xCoords, &minX, &maxX);
		cv::minMaxLoc(yCoords, &minY, &maxY);
		float faceboxScaleFactor = 1.25f;
		float modelWidth = maxX - minX;
		float modelHeight = maxY - minY;
		// scale it:
		modelShape = modelShape * (faces[0].width / modelWidth + faces[0].height / modelHeight) / (2.0f * faceboxScaleFactor);
		// translate the model:
		Scalar meanX = cv::mean(xCoords);
		double meanXd = meanX[0];
		Scalar meanY = cv::mean(yCoords);
		double meanYd = meanY[0];
		// move it:
		xCoords += faces[0].x + faces[0].width / 2.0f - meanXd;
		yCoords += faces[0].y + faces[0].height / 2.0f - meanYd;

		for (int i = 0; i < m.getNumLandmarks(); ++i) {
			cv::circle(landmarksImage, Point2f(modelShape.at<float>(i, 0), modelShape.at<float>(i + m.getNumLandmarks(), 0)), 3, Scalar(255.0f, 0.0f, 255.0f));
		}

		for (int hogScale = 0; hogScale < m.getNumHogScales(); ++hogScale) {
			//feature_current = obtain_features(double(TestImg), New_Shape, 'HOG', hogScale);
			int cellSize, numBins;
			switch (hogScale) // should go into the model or a "hogOptimizer"
			{
			case 0:
				cellSize = 3;
				numBins = 4;
				break;
			case 1:
				cellSize = 3;
				numBins = 4;
				break;
			case 2:
				cellSize = 2;
				numBins = 4;
				break;
			case 3:
				cellSize = 2;
				numBins = 4;
				break;
			case 4:
				cellSize = 1;
				numBins = 4;
				break;
			default:
				break; // should never happen
			}
			int numNeighbours = cellSize * 6; // this cellSize has nothing to do with HOG. It's the number of "cells", i.e. image-windows/patches.
											  // if cellSize=1, our window is 12x12, and because our HOG-cellsize is 12, it means we will have 1 cell (the minimum).
			int hogCellSize = 12;
			int hogDim1 = (numNeighbours * 2) / hogCellSize; // i.e. how many times does the hogCellSize fit into our patch
			int hogDim2 = hogDim1; // as our patch is quadratic, those two are the same
			int hogDim3 = 16; // I don't know yet where this comes from, maybe numOrientations*numOrientations?
			int hogDims = hogDim1 * hogDim2 * hogDim3;
			Mat currentFeatures(m.getNumLandmarks() * hogDims, 1, CV_32FC1);

			for (int i = 0; i < m.getNumLandmarks(); ++i) {
				// get the (x, y) location and w/h of the current patch
				int x = cvRound(modelShape.at<float>(i, 0));
				int y = cvRound(modelShape.at<float>(i+m.getNumLandmarks(), 0));
				cv::Rect roi(x, y, numNeighbours * 2, numNeighbours * 2); // x y w h
				// extract the patch and supply it to vl_hog
				Mat roiImg = imgGray(roi).clone(); // clone because we need a continuous memory block
				roiImg.convertTo(roiImg, CV_32FC1); // because vl_hog_put_image expects a float* (values 0.f-255.f)
				VlHog* hog = vl_hog_new(VlHogVariant::VlHogVariantUoctti, /*numOrientations=*/numBins, true); // VlHogVariantUoctti seems to be default in Matlab
				vl_hog_put_image(hog, (float*)roiImg.data, roiImg.cols, roiImg.rows, /*numChannels=*/1, hogCellSize);
				vl_size ww = vl_hog_get_width(hog);
				vl_size hh = vl_hog_get_height(hog);
				vl_size dd = vl_hog_get_dimension(hog); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
				float* hogArray = (float*)vl_malloc(ww*hh*dd*sizeof(float));
				vl_hog_extract(hog, hogArray);
				vl_hog_delete(hog);
				Mat hogFeatures(ww*hh*dd, 1, CV_32FC1, hogArray);

				//features = [features; double(reshape(tmp, [], 1))];
				// B = reshape(A,m,n) returns the m-by-n matrix B whose elements are taken column-wise from A
				Mat currentFeaturesSubrange = currentFeatures.rowRange(i * hogDims, i * hogDims + hogDims);
				hogFeatures.copyTo(currentFeaturesSubrange);
				// currentFeatures needs to have dimensions n x 1, where n = numLandmarks * hogFeaturesDimension, e.g. n = 22 * (3*3*16=144) = 3168 (for the first hog Scale)
			}
			

			//delta_shape = AAM.RF(1).Regressor(hogScale).A(1:end - 1, : )' * feature_current + AAM.RF(1).Regressor(hogScale).A(end,:)';
			Mat regressorData = m.getRegressorData(hogScale);
			Mat deltaShape = regressorData.rowRange(0, regressorData.rows - 1).t() * currentFeatures + regressorData.row(regressorData.rows - 1).t();

			modelShape = modelShape + deltaShape;
			/*
			for (int i = 0; i < m.getNumLandmarks(); ++i) {
				cv::circle(landmarksImage, Point2f(modelShape.at<float>(i, 0), modelShape.at<float>(i + m.getNumLandmarks(), 0)), 6 - hogScale, Scalar(51.0f*(float)hogScale, 51.0f*(float)hogScale, 0.0f));
			}*/

		}
		
		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");
		
		cv::imshow(windowName, landmarksImage);
		cv::waitKey();

	}

	return 0;
}
