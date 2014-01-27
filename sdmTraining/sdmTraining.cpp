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
	//#include "vl/hog.h"
	#include "hog.h"
}

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/nonfree/nonfree.hpp"

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
#include "imageio/CameraImageSource.hpp"
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

/*
Some notes:
 - The current model ('SDM_Model_HOG_Zhenhua_11012014.txt') uses roughly 1/10 of
   the training data of the original model from the paper, and has no expressions

 - One problem: Running the optimization several times doesn't result in better
   performance. Two possible reasons:
     * In the training, what we train is the step from the mean to the groundtruth.
	   So we only train a big step.
	     - Actually, that means that it's very important to get the rigid alignment
		   right to get the first update-step right?
	 * The update-step for one landmark is dependent on the other landmarks

 Test: To calculate the face-box (Zhenhua): Take all 68 LMs; Take the min/max x and y
 for the face-box. (so the face-box is quite small)
*/
class HogSdmModel
{
public:
	HogSdmModel() {
	};

	struct HogParameter
	{
		int cellSize;
		int numBins;
	};

	// interface
	int getNumLandmarks() const {
		return meanLandmarks.rows/2;
	};

	// only HOG models
	int getNumHogScales() const {
		return regressorData.size();
	};

	HogParameter getHogParameters(int hogScaleLevel) {
		return hogParameters[hogScaleLevel];
	}

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
			HogParameter params;
			params.cellSize = lexical_cast<int>(stringContainer[7]); // = cellSize
			params.numBins = lexical_cast<int>(stringContainer[9]); // = numBins
			model.hogParameters.push_back(params);
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

	std::vector<HogParameter> hogParameters;

};

class HogSdmModelFitting
{
public:
	HogSdmModelFitting(HogSdmModel model)/* : model(model)*/ {
		this->model = model;
	};
	
	// out: aligned modelShape
	// in: Rect, ocv with tl x, tl y, w, h (?) and calcs center
	// directly modifies modelShape
	// could move to parent-class
	cv::Mat alignRigid(cv::Mat modelShape, cv::Rect faceBox) const {
		
		Mat xCoords = modelShape.rowRange(0, modelShape.rows / 2);
		Mat yCoords = modelShape.rowRange(modelShape.rows / 2, modelShape.rows);
		// scale the model:
		double minX, maxX, minY, maxY;
		cv::minMaxLoc(xCoords, &minX, &maxX);
		cv::minMaxLoc(yCoords, &minY, &maxY);
		float faceboxScaleFactor = 1.25f; // 1.25f: value of Zhenhua Matlab FD. Mine: 1.35f
		float modelWidth = maxX - minX;
		float modelHeight = maxY - minY;
		// scale it:
		modelShape = modelShape * (faceBox.width / modelWidth + faceBox.height / modelHeight) / (2.0f * faceboxScaleFactor);
		// translate the model:
		Scalar meanX = cv::mean(xCoords);
		double meanXd = meanX[0];
		Scalar meanY = cv::mean(yCoords);
		double meanYd = meanY[0];
		// move it:
		xCoords += faceBox.x + faceBox.width / 2.0f - meanXd;
		yCoords += faceBox.y + faceBox.height / 1.8f - meanYd; // we use another value for y because we don't want to center the model right in the middle of the face-box

		return modelShape;
	};

	// out: optimized model-shape
	// in: GRAY img
	// in: evtl zusaetzlicher param um scale-level/iter auszuwaehlen
	// calculates shape updates (deltaShape) for one or more iter/scales and returns...
	cv::Mat optimize(cv::Mat modelShape, cv::Mat image) {
		
		for (int hogScale = 0; hogScale < model.getNumHogScales(); ++hogScale) {
			//feature_current = obtain_features(double(TestImg), New_Shape, 'HOG', hogScale);
			HogSdmModel::HogParameter hogParameter = model.getHogParameters(hogScale);
			int numNeighbours = hogParameter.cellSize * 6; // this cellSize has nothing to do with HOG. It's the number of "cells", i.e. image-windows/patches.
			// if cellSize=1, our window is 12x12, and because our HOG-cellsize is 12, it means we will have 1 cell (the minimum).
			int hogCellSize = 12;
			int hogDim1 = (numNeighbours * 2) / hogCellSize; // i.e. how many times does the hogCellSize fit into our patch
			int hogDim2 = hogDim1; // as our patch is quadratic, those two are the same
			int hogDim3 = 16; // VlHogVariantUoctti: Creates 4+3*numOrientations dimensions
			int hogDims = hogDim1 * hogDim2 * hogDim3;
			Mat currentFeatures(model.getNumLandmarks() * hogDims, 1, CV_32FC1);

			for (int i = 0; i < model.getNumLandmarks(); ++i) {
				// get the (x, y) location and w/h of the current patch
				int x = cvRound(modelShape.at<float>(i, 0));
				int y = cvRound(modelShape.at<float>(i + model.getNumLandmarks(), 0));
				cv::Rect roi(x - numNeighbours, y - numNeighbours, numNeighbours * 2, numNeighbours * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
				// we have exactly the same window as the matlab code.
				// extract the patch and supply it to vl_hog
				Mat roiImg = image(roi).clone(); // clone because we need a continuous memory block
				roiImg.convertTo(roiImg, CV_32FC1); // because vl_hog_put_image expects a float* (values 0.f-255.f)
				VlHog* hog = vl_hog_new(VlHogVariant::VlHogVariantUoctti, /*numOrientations=*/hogParameter.numBins, /*transposed (=col-major):*/false); // VlHogVariantUoctti seems to be default in Matlab.
				vl_hog_put_image(hog, (float*)roiImg.data, roiImg.cols, roiImg.rows, /*numChannels=*/1, hogCellSize);
				vl_size ww = vl_hog_get_width(hog);
				vl_size hh = vl_hog_get_height(hog);
				vl_size dd = vl_hog_get_dimension(hog); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
				//float* hogArray = (float*)malloc(ww*hh*dd*sizeof(float));
				Mat hogArray(1, ww*hh*dd, CV_32FC1); // safer & same result. Don't use C-style memory management.
				//vl_hog_extract(hog, hogArray); // just interpret hogArray in col-major order to get the same n x 1 vector as in matlab. (w * h * d)
				vl_hog_extract(hog, hogArray.ptr<float>(0));
				vl_hog_delete(hog);
				Mat hogDescriptor(hh*ww*dd, 1, CV_32FC1);
				for (int j = 0; j < dd; ++j) {
					//Mat hogFeatures(hh, ww, CV_32FC1, hogArray + j*ww*hh);
					Mat hogFeatures(hh, ww, CV_32FC1, hogArray.ptr<float>(0) + j*ww*hh); // Creates the same array as in Matlab. I might have to check this again if hh!=ww (non-square)
					hogFeatures = hogFeatures.t(); // Necessary because the Matlab reshape() takes column-wise from the matrix while the OpenCV reshape() takes row-wise.
					hogFeatures = hogFeatures.reshape(0, hh*ww); // make it to a column-vector
					Mat currentDimSubMat = hogDescriptor.rowRange(j*ww*hh, j*ww*hh + ww*hh);
					hogFeatures.copyTo(currentDimSubMat);
				}
				//free(hogArray); // not necessary - we use a Mat.
				//features = [features; double(reshape(tmp, [], 1))];
				// B = reshape(A,m,n) returns the m-by-n matrix B whose elements are taken column-wise from A
				// Matlab (& Eigen, OpenGL): Column-major.
				// OpenCV: Row-major.
				// (access is always (r, c).)
				Mat currentFeaturesSubrange = currentFeatures.rowRange(i*hogDims, i*hogDims + hogDims);
				hogDescriptor.copyTo(currentFeaturesSubrange);
				// currentFeatures needs to have dimensions n x 1, where n = numLandmarks * hogFeaturesDimension, e.g. n = 22 * (3*3*16=144) = 3168 (for the first hog Scale)
			}

			//delta_shape = AAM.RF(1).Regressor(hogScale).A(1:end - 1, : )' * feature_current + AAM.RF(1).Regressor(hogScale).A(end,:)';
			Mat regressorData = model.getRegressorData(hogScale);
			Mat deltaShape = regressorData.rowRange(0, regressorData.rows - 1).t() * currentFeatures + regressorData.row(regressorData.rows - 1).t();

			modelShape = modelShape + deltaShape;
			/*
			for (int i = 0; i < m.getNumLandmarks(); ++i) {
			cv::circle(landmarksImage, Point2f(modelShape.at<float>(i, 0), modelShape.at<float>(i + m.getNumLandmarks(), 0)), 6 - hogScale, Scalar(51.0f*(float)hogScale, 51.0f*(float)hogScale, 0.0f));
			}*/
		}

		return modelShape;
	};

private:
	HogSdmModel model;
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
	vector<path> inputPaths;
	path inputFilelist;
	path inputDirectory;
	vector<path> inputFilenames;
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
			("input,i", po::value<vector<path>>(&inputPaths)->required(), 
				"input from one or more files, a directory, or a  .lst/.txt-file containing a list of images")
			("landmarks,l", po::value<path>(&landmarksDir)->required(), 
				"load landmark files from the given folder")
			("landmark-type,t", po::value<string>(&landmarkType)->required(), 
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
	Loggers->getLogger("sdmTraining").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("sdmTraining");

	appLogger.debug("Verbose level for console output: " + logging::loglevelToString(logLevel));

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

	vector<path> groundtruthDirs; groundtruthDirs.push_back(landmarksDir); // Todo: Make cmdline use a vector<path>
	shared_ptr<LandmarkFormatParser> landmarkFormatParser;
	if (boost::iequals(landmarkType, "ibug")) {
		landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
		landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, ".pts", GatherMethod::ONE_FILE_PER_IMAGE_SAME_DIR, groundtruthDirs), landmarkFormatParser);
	} else {
		cout << "Error: Invalid ground truth type." << endl;
		return EXIT_FAILURE;
	}

	labeledImageSource = make_shared<NamedLabeledImageSource>(imageSource, landmarkSource);
		
	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	const string windowName = "win";

	vector<imageio::ModelLandmark> landmarks;

	cv::namedWindow(windowName);

	//HogSdmModel hogModel = HogSdmModel::load("C:\\Users\\Patrik\\Documents\\GitHub\\SGD_Zhenhua_11012014\\SDM_Model_HOG_Zhenhua_11012014.txt");
	//HogSdmModelFitting modelFitter(hogModel);

	string faceDetectionModel("C:\\opencv\\2.4.7.2_prebuilt\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"); // sgd: "../models/haarcascade_frontalface_alt2.xml"
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(faceDetectionModel))
	{
		cout << "Error loading face detection model." << endl;
		return EXIT_FAILURE;
	}

	// 1. First, check on which faces the face-detection succeeds. Then only use these images for training.
	//    "succeeds" means all groundtruth landmarks are inside the face-box. This is reasonable for a face-
	//    detector like OCV V&J which has a big face-box, but for others, another method is necessary.
	vector<std::tuple<Mat, cv::Rect, vector<shared_ptr<Landmark>>>> trainingData;
	while (labeledImageSource->next()) {
		vector<shared_ptr<Landmark>> landmarks = labeledImageSource->getLandmarks().getLandmarks();
		// check if landmarks.size() == numModelLandmarks

		Mat img = labeledImageSource->getImage();
		vector<cv::Rect> detectedFaces;
		faceCascade.detectMultiScale(img, detectedFaces, 1.2, 2, 0, cv::Size(50, 50));
		if (detectedFaces.empty()) {
			continue;
		}
		Mat output = img.clone();
		for (const auto& f : detectedFaces) {
			cv::rectangle(output, f, cv::Scalar(0.0f, 0.0f, 255.0f));
		}
		// check if the detected face is a valid one:
		// i.e. for now, if the groundtruth landmarks 37 (reye_oc), 46 (leye_oc) and 58 (mouth_ll_c) are inside the face-box
		// (should add: _and_ the face-box is not bigger than IED*2 or something)
		bool skipImage = false;
		for (const auto& lm : landmarks) {
			if (lm->getName() == "37" || lm->getName() == "46" || lm->getName() == "58") {
				if (!detectedFaces[0].contains(lm->getPoint2D())) {
					skipImage = true;
					break; // if any LM is not inside, skip this training image
					// Note: improvement: if the first facebox doesn't work, try the other ones
				}
			}
		}
		if (skipImage) {
			continue;
		}
		trainingData.push_back(std::make_tuple(img, detectedFaces[0], landmarks));
	}

	// 2. Calculate the mean-shape of all training images
	//    Afterwards: We could do some procrustes and align all shapes to the calculated mean-shape. But actually just the mean calculated above is a good approximation.
	//    Q: At least do some centering/scaling?
	int numImages = trainingData.size();
	int numModelLandmarks = 68; // Todo: Do dynamically?
	int numSamplesPerImage = 2; // How many Monte Carlo samples to generate per training image
	Mat groundtruthLandmarks(2 * numModelLandmarks, numImages, CV_32FC1);
	Mat groundtruthLandmarksNormalizedByUnitFacebox(2 * numModelLandmarks, numImages, CV_32FC1);
	int currentImage = 0;
	for (const auto& data : trainingData) {
		Mat img = std::get<0>(data);
		cv::Rect detectedFace = std::get<1>(data);
		vector<shared_ptr<Landmark>> lmsv = std::get<2>(data);
		// check if lmsv.size() == numModelLandmarks
		
		for (int i = 0; i < lmsv.size(); ++i) {
			float normalizedX = ((lmsv[i]->getX() - detectedFace.x) / static_cast<float>(detectedFace.width)) - 0.5f;
			float normalizedY = ((lmsv[i]->getY() - detectedFace.y) / static_cast<float>(detectedFace.height)) - 0.5f;
			groundtruthLandmarksNormalizedByUnitFacebox.at<float>(i, currentImage) = normalizedX;
			groundtruthLandmarksNormalizedByUnitFacebox.at<float>(i + numModelLandmarks, currentImage) = normalizedY;
			groundtruthLandmarks.at<float>(i, currentImage) = lmsv[i]->getX();
			groundtruthLandmarks.at<float>(i + numModelLandmarks, currentImage) = lmsv[i]->getY();
		}
		currentImage++;
	}
	// Take the mean of every row:
	Mat modelMean;
	cv::reduce(groundtruthLandmarksNormalizedByUnitFacebox, modelMean, 1, CV_REDUCE_AVG); // reduce to 1 column
	
	std::ofstream myfile;
	myfile.open("example.txt");
	myfile << "x = [";
	for (int i = 0; i < numModelLandmarks; ++i) {
		myfile << modelMean.at<float>(i, 0) << ", ";
	}
	myfile << "];" << std::endl << "y = [";
	for (int i = 0; i < numModelLandmarks; ++i) {
		myfile << modelMean.at<float>(i + numModelLandmarks, 0) << ", ";
	}
	myfile << "];" << std::endl;
	myfile.close();
	// Q: Why is our mean twice as big? Their mean doesn't fit into a V&J facebox from -0.5 to 0.5?

	/*
	Mat test = Mat::zeros(480, 640, CV_8UC3);
	for (int i = 0; i < numModelLandmarks; ++i) {
		cv::circle(test, Point2f(groundtruthLandmarksMean.at<float>(i, 0), groundtruthLandmarksMean.at<float>(i + numModelLandmarks, 0)), 3, Scalar(0.0f, 0.0f, 255.0f));
	}*/

	// 2.9 Calculate the mean and variances of the translational and scaling differences between the initial and true landmark locations. (used for generating the samples)
	// At the same time, store the initial shape estimate (x_0)
	float muT = 4.0f;
	float sigmaT = 1.0f;
	float muS = 1.0f;
	float sigmaS = 0.07f;
	Mat initialShape = Mat::zeros(2 * numModelLandmarks, (numSamplesPerImage+1)* trainingData.size(), CV_32FC1); // 10 samples + the original data = 11
	currentImage = 0;
	for (const auto& data : trainingData) {
		Mat img = std::get<0>(data);
		cv::Rect detectedFace = std::get<1>(data);
		vector<shared_ptr<Landmark>> lmsv = std::get<2>(data);
		
		// Initial esti: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
		Mat initialShapeEstimateX0 = modelMean.clone();
		Mat initialShapeEstimateX0_x = initialShapeEstimateX0.rowRange(0, initialShapeEstimateX0.rows / 2);
		Mat initialShapeEstimateX0_y = initialShapeEstimateX0.rowRange(initialShapeEstimateX0.rows / 2, initialShapeEstimateX0.rows);
		initialShapeEstimateX0_x = (initialShapeEstimateX0_x + 0.5f) * detectedFace.width + detectedFace.x;
		initialShapeEstimateX0_y = (initialShapeEstimateX0_y + 0.5f) * detectedFace.height + detectedFace.y;
		Mat initialShapeColumn = initialShape.col(currentImage * (numSamplesPerImage+1));
		initialShapeEstimateX0.copyTo(initialShapeColumn);
		for (int i = 0; i < numModelLandmarks; ++i) {
			cv::circle(img, Point2f(initialShapeEstimateX0.at<float>(i, 0), initialShapeEstimateX0.at<float>(i + numModelLandmarks, 0)), 3, Scalar(0.0f, 255.0f, 0.0f));
		}

		// Calculate the mean and variances of the translational and scaling differences between the initial and true landmark locations:
		// Q: how?
		currentImage++;
	}

	// 3. For every training image:
	std::mt19937 engine; ///< A Mersenne twister MT19937 engine
	engine.seed();
	std::normal_distribution<float> rndNormalTranslation(muT, sigmaT); // second param is stddev.
	std::normal_distribution<float> rndNormalScaling(muS, sigmaS); // second param is stddev.
	currentImage = 0;
	for (const auto& data : trainingData) {
		// a) Run the face-detector (the same that we're going to use in the testing-stage) (already done)
		Mat img = std::get<0>(data);
		cv::Rect detectedFace = std::get<1>(data);
		vector<shared_ptr<Landmark>> lmsv = std::get<2>(data);
		// b) Align the model? Rigid alignment? (only centering of the mean). x_0
		// Already done in previous step, stored in 'initialShape'

		// c) Generate Monte Carlo samples? With what variance? x_0^i (maybe make this step 3.)
		// sample around initialShapeThis, store in initialShape
		//		Save the samples, all in a matrix
		//		Todo 1) don't use pixel variance, but a scale-independent one (normalize by IED?)
		//			 2) calculate the variance from data (gt facebox?)
		for (int i = 0; i < numSamplesPerImage; ++i) {
			float translation = rndNormalTranslation(engine);
			float scaling = rndNormalScaling(engine);
			Mat currentSample = initialShape.col(currentImage*(numSamplesPerImage+1) + (i+1));
			initialShape.col(currentImage*(numSamplesPerImage+1)).copyTo(currentSample);
			currentSample = currentSample * scaling + translation; // Check if the sample goes outside the feature-extractable region?
			// TODO: The scaling needs to be done in the normalized facebox region?? Try to write it down?
			// Better do the translation in the norm-FB as well to be independent of face-size?
		}
		currentImage++;
	}
	// 4. For every sample plus the original image: (loop through the matrix)
	//			a) groundtruthShape, initialShape
	//				deltaShape = ground - initial
	Mat groundtruthShapes = Mat::zeros(2 * numModelLandmarks, (numSamplesPerImage+1)* trainingData.size(), CV_32FC1); // 10 samples + the original data = 11
	for (int currImg = 0; currImg < groundtruthLandmarks.cols; ++currImg) {
		Mat groundtruthLandmarksCol = groundtruthLandmarks.col(currImg);
		for (int j = 0; j < numSamplesPerImage + 1; ++j) {
			Mat groundtruthShapesCol = groundtruthShapes.col(currImg*(numSamplesPerImage + 1) + j);
			groundtruthLandmarksCol.copyTo(groundtruthShapesCol);
		}
	}
	Mat deltaShape = groundtruthShapes - initialShape;
	//			b) Extract the features at all landmark locations initialShape (Paper: SIFT, 32x32 (?))
	int featureDimension = 128;
	Mat featureMatrix = Mat::ones((featureDimension * numModelLandmarks) + 1, initialShape.cols, CV_32FC1); // Our 'A'. The last row stays all 1's; it's for learning the offset/bias
	currentImage = 0;
	for (const auto& data : trainingData) {
		Mat img = std::get<0>(data);
		//cv::Rect detectedFace = std::get<1>(data);
		//vector<shared_ptr<Landmark>> lmsv = std::get<2>(data);
		Mat imgGray; // img delete fb
		cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
		cv::SIFT sift;
		for (int sample = 0; sample < numSamplesPerImage + 1; ++sample) {
			vector<cv::KeyPoint> keypoints;
			for (int lm = 0; lm < numModelLandmarks; ++lm) {
				float px = initialShape.at<float>(lm, currentImage*(numSamplesPerImage + 1) + sample);
				float py = initialShape.at<float>(lm + numModelLandmarks, currentImage*(numSamplesPerImage + 1) + sample);
				keypoints.push_back(cv::KeyPoint(px, py, 32.0f, 0.0f)); // Angle is set to 0. If it's -1, SIFT will be calculated for 361degrees.
			}
			Mat siftDescriptors;
			sift(imgGray, Mat(), keypoints, siftDescriptors, true);
			cv::drawKeypoints(img, keypoints, img, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			// concatenate all the descriptors for this sample vertically (into a column-vector)
			siftDescriptors = siftDescriptors.reshape(0, featureDimension * numModelLandmarks);
			int currentImageMatrixIndex = currentImage*(numSamplesPerImage + 1) + sample;
			//			c) store the features
			Mat featureMatrixCol = featureMatrix.col(currentImageMatrixIndex).rowRange(0, featureDimension * numModelLandmarks);
			siftDescriptors.copyTo(featureMatrixCol);
		}
		++currentImage;
	}
	
	// 5. Add one row to the features (already done), add regLambda
	float lambda = 10.0f;
	Mat regulariser = Mat::eye(featureDimension * numModelLandmarks + 1, featureDimension * numModelLandmarks + 1, CV_32FC1) * lambda;
	regulariser.at<float>(regulariser.rows - 1, regulariser.cols - 1) = 0.0f; // no lambda for the bias
	//		solve for x!
	Mat AAt = featureMatrix * featureMatrix.t();
	Mat AAtReg = AAt + regulariser;
	Mat AAtRegInv = AAtReg.inv(/*cv::DECOMP_SVD*/);
	Mat AAtRegInvA = AAtRegInv * featureMatrix;
	Mat AAtRegInvAbt = AAtRegInvA * deltaShape.t(); // = x

	// 6. Do the same again for all cascade steps

	while(labeledImageSource->next()) {
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + labeledImageSource->getName().string());
		img = labeledImageSource->getImage();
		
		LandmarkCollection lms = labeledImageSource->getLandmarks();
		vector<shared_ptr<Landmark>> lmsv = lms.getLandmarks();
		landmarks.clear();
		Mat landmarksImage = img.clone(); // blue rect = the used landmarks
		/*
		for (const auto& lm : lmsv) {
			lm->draw(landmarksImage);
			landmarks.emplace_back(imageio::ModelLandmark(lm->getName(), lm->getPosition2D()));
			cv::rectangle(landmarksImage, cv::Point(cvRound(lm->getX() - 2.0f), cvRound(lm->getY() - 2.0f)), cv::Point(cvRound(lm->getX() + 2.0f), cvRound(lm->getY() + 2.0f)), cv::Scalar(255, 0, 0));
		}*/

		Mat imgGray;
		cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
		vector<cv::Rect> faces;
		float score, notFace = 0.5;
		
		// face detection
		//faceCascade.detectMultiScale(img, faces, 1.2, 2, 0, cv::Size(50, 50));
		faces.push_back({ 172, 199, 278, 278 });
		if (faces.empty()) {
			cv::imshow(windowName, landmarksImage);
			cv::waitKey(5);
			continue;
		}
		for (const auto& f : faces) {
			cv::rectangle(landmarksImage, f, cv::Scalar(0.0f, 0.0f, 255.0f));
		}
		
		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");
		
		cv::imshow(windowName, landmarksImage);
		cv::waitKey(5);

	}

	return 0;
}
