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

#include "shapemodels/SdmLandmarkModel.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
using namespace shapemodels;
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

enum class MeanTraining { // not needed anymore
	// on what to train the mean:
	ON_VJ_DETECTIONS, // orig paper, sect. 3.1. & Fig. 2b)
	ON_ALL // orig paper?
};

enum class MeanPreAlign { // what to do with the GT LMs before mean is taken.
	NONE // no prealign, stay in img-coords
	// translate/scale to normalized square (?)
};

enum class MeanNormalization { // what to do with the mean coords after the mean has been calculated
	OLD_METHOD, // NORMALIZE_BY_FACEBOX?
	UNIT_SUM_SQUARED_NORMS // orig paper?
};
// computation...normalization ... alignment ...

enum class FilterByFaceDetection {
	NONE,
	VIOLAJONES
};

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path outputFilename;
	path configFilename;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("config,c", po::value<path>(&configFilename)->required(),
				"input config")
			("output,o", po::value<path>(&outputFilename)->required(),
				"output")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << "Usage: sdmTraining [options]\n";
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

	struct Dataset {
		string databaseName;
		path images;
		path groundtruth;
		string landmarkType;
		map<string, string> landmarkMappings; // from model (lhs) to thisDb (rhs)

		shared_ptr<LabeledImageSource> labeledImageSource;
	};

	string modelLandmarkType;
	std::vector<string> modelLandmarks;
	vector<Dataset> trainingDatasets;
	int numSamplesPerImage; // How many Monte Carlo samples to generate per training image, in addition to the original image. Default: 10

	// Read the stuff from the config:
	ptree pt;
	try {
		read_info(configFilename.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	try {
		// Get stuff from the parameters subtree
		ptree ptParameters = pt.get_child("parameters");
		numSamplesPerImage = ptParameters.get<int>("numSamplesPerImage", 10);

		// Get stuff from the modelLandmarks subtree:
		ptree ptModelLandmarks = pt.get_child("modelLandmarks");
		modelLandmarkType = ptModelLandmarks.get<string>("landmarkType");
		appLogger.debug("Type of the model landmarks: " + modelLandmarkType);
		string modelLandmarksUsage = ptModelLandmarks.get<string>("landmarks");
		if (modelLandmarksUsage.empty()) {
			// value is empty, meaning it's a node and the user should specify a list of 'landmarks'
			ptree ptmodelLandmarksList = ptModelLandmarks.get_child("landmarks");
			for (const auto& kv : ptmodelLandmarksList) {
				modelLandmarks.push_back(kv.first);
			}
			appLogger.debug("Loaded a list of " + lexical_cast<string>(modelLandmarks.size()) + " landmarks to train the model.");
		}
		else if (modelLandmarksUsage == "all") {
			throw std::logic_error("Using 'all' modelLandmarks is not implemented yet - specify a list for now.");
		} 
		else {
			throw std::logic_error("Error reading the models 'landmarks' key, should either provide a node with a list of landmarks or specify 'all'.");
		}

		// Get stuff from the trainingData subtree:
		ptree ptTrainingData = pt.get_child("trainingData");
		for (const auto& kv : ptTrainingData) { // For each database:
			appLogger.debug("Using database '" + kv.first + "' for training:");
			Dataset dataset;
			dataset.databaseName = kv.first;
			dataset.images = kv.second.get<path>("images");
			dataset.groundtruth = kv.second.get<path>("groundtruth");
			dataset.landmarkType = kv.second.get<string>("landmarkType");
			string landmarkMappingsUsage = kv.second.get<string>("landmarkMappings");
			if (landmarkMappingsUsage.empty()) {
				// value is empty, meaning it's a node and the user should specify a list of landmarkMappings
				ptree ptLandmarkMappings = kv.second.get_child("landmarkMappings");
				for (const auto& mapping : ptLandmarkMappings) {
					dataset.landmarkMappings.insert(make_pair(mapping.first, mapping.second.get_value<string>()));
				}
				appLogger.debug("Loaded a list of " + lexical_cast<string>(dataset.landmarkMappings.size()) + " landmark mappings.");
				if (dataset.landmarkMappings.size() < modelLandmarks.size()) {
					throw std::logic_error("Error reading the landmark mappings, there are less mappings given than the number of landmarks that should be used to train the model.");
				}
			}
			else if (landmarkMappingsUsage == "none") {
				// generate identity mappings
				for (const auto& lm : modelLandmarks) {
					dataset.landmarkMappings.insert(make_pair(lm, lm));
				}
				appLogger.debug("Generated a list of " + lexical_cast<string>(dataset.landmarkMappings.size()) + " identity landmark mappings.");
			}
			else {
				throw std::logic_error("Error reading the landmark mappings, should either provide list of mappings or specify 'none'.");
			}
			trainingDatasets.push_back(dataset);
		}
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error("Parsing config: " + string(error.what()));
		return EXIT_FAILURE;
	}
	catch (const std::logic_error& e) {
		appLogger.error("Parsing config: " + string(e.what()));
		return EXIT_FAILURE;
	}

	// Read in the image sources:
	for (auto& d : trainingDatasets) {
		// Load the images:
		shared_ptr<ImageSource> imageSource;
		// We assume the user has given either a directory or a .lst/.txt-file
		if (d.images.extension().string() == ".lst" || d.images.extension().string() == ".txt") { // check for .lst or .txt first
			appLogger.info("Using file-list as input: " + d.images.string());
			shared_ptr<ImageSource> fileListImgSrc; // TODO VS2013 change to unique_ptr, rest below also
			try {
				fileListImgSrc = make_shared<FileListImageSource>(d.images.string());
			}
			catch (const std::runtime_error& e) {
				appLogger.error(e.what());
				return EXIT_FAILURE;
			}
			imageSource = fileListImgSrc;
		}
		else if (boost::filesystem::is_directory(d.images)) {
			appLogger.info("Using input images from directory: " + d.images.string());
			try {
				imageSource = make_shared<DirectoryImageSource>(d.images.string());
			}
			catch (const std::runtime_error& e) {
				appLogger.error(e.what());
				return EXIT_FAILURE;
			}
		}
		else {
			appLogger.error("The path given is neither a directory nor a .lst/.txt-file containing a list of images.");
			return EXIT_FAILURE;
		}
		// Load the ground truth
		shared_ptr<NamedLandmarkSource> landmarkSource;
		vector<path> groundtruthDirs = { d.groundtruth };
		shared_ptr<LandmarkFormatParser> landmarkFormatParser;
		if (boost::iequals(d.landmarkType, "ibug")) {
			landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
			landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, ".pts", GatherMethod::ONE_FILE_PER_IMAGE_SAME_DIR, groundtruthDirs), landmarkFormatParser);
		}
		else {
			cout << "Error: Invalid ground truth type." << endl;
			return EXIT_FAILURE;
		}
		d.labeledImageSource = make_shared<NamedLabeledImageSource>(imageSource, landmarkSource);
	}

		
	std::chrono::time_point<std::chrono::system_clock> start, end;

	//vector<imageio::ModelLandmark> landmarks;

	//HogSdmModel hogModel = HogSdmModel::load("C:\\Users\\Patrik\\Documents\\GitHub\\SGD_Zhenhua_11012014\\SDM_Model_HOG_Zhenhua_11012014.txt");
	//HogSdmModelFitting modelFitter(hogModel);

	string faceDetectionModel("C:\\opencv\\2.4.7.2_prebuilt\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"); // sgd: "../models/haarcascade_frontalface_alt2.xml"
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(faceDetectionModel))
	{
		cout << "Error loading face detection model." << endl;
		return EXIT_FAILURE;
	}
	// Settings:
	//MeanTraining meanTraining = MeanTraining::ON_VJ_DETECTIONS;
	MeanPreAlign meanPreAlign = MeanPreAlign::NONE;
	MeanNormalization meanNormalization = MeanNormalization::UNIT_SUM_SQUARED_NORMS;
	FilterByFaceDetection filterByFaceDetection = FilterByFaceDetection::VIOLAJONES;
	// Add a switch "use_GT_as_detectionResults" ?
	
	// Our data:
	vector<std::tuple<Mat, cv::Rect, Mat>> trainingData; // img, fb, groundtruth-lms (model ones)
	for (const auto& d : trainingDatasets) {
		if (filterByFaceDetection == FilterByFaceDetection::NONE) {
			// 1. Use all the images for training
			while (d.labeledImageSource->next()) {
				Mat landmarks(1, 2 * modelLandmarks.size(), CV_32FC1);
				int currentLandmark = 0;
				for (const auto ml : modelLandmarks) {
					try {
						string lmIdInDb = d.landmarkMappings.at(ml);
						shared_ptr<Landmark> lm = d.labeledImageSource->getLandmarks().getLandmark(lmIdInDb);
						landmarks.at<float>(0, currentLandmark) = lm->getX();
						landmarks.at<float>(0, currentLandmark + modelLandmarks.size()) = lm->getY();
					} catch (std::out_of_range& e) {
						appLogger.error(e.what()); // mapping failed
						return EXIT_FAILURE;
					}
					catch (std::invalid_argument& e) {
						appLogger.error(e.what()); // lm not in db
						return EXIT_FAILURE;
					}
					++currentLandmark;
				}
				Mat img = d.labeledImageSource->getImage();
				trainingData.push_back(std::make_tuple(img, cv::Rect(), landmarks));
			}
		}
		else if (filterByFaceDetection == FilterByFaceDetection::VIOLAJONES) {
			// 1. First, check on which faces the face-detection succeeds. Then only use these images for training.
			//    "succeeds" means all ground-truth landmarks are inside the face-box. This is reasonable for a face-
			//    detector like OCV V&J which has a big face-box, but for others, another method is necessary.
			while (d.labeledImageSource->next()) {
				Mat img = d.labeledImageSource->getImage();
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
				// i.e. for now, if the ground-truth landmarks 37 (reye_oc), 46 (leye_oc) and 58 (mouth_ll_c) are inside the face-box
				// (should add: _and_ the face-box is not bigger than IED*2 or something)
				vector<shared_ptr<Landmark>> allLandmarks = d.labeledImageSource->getLandmarks().getLandmarks();
				bool skipImage = false;
				for (const auto& lm : allLandmarks) {
					if (lm->getName() == "37" || lm->getName() == "46" || lm->getName() == "58") {
						if (!detectedFaces[0].contains(lm->getPoint2D())) {
							skipImage = true;
							break; // if any LM is not inside, skip this training image
							// Note: improvement: if the first face-box doesn't work, try the other ones
						}
					}
				}
				if (skipImage) {
					continue;
				}
				// We're using the image:
				Mat landmarks(1, 2 * modelLandmarks.size(), CV_32FC1);
				int currentLandmark = 0;
				for (const auto ml : modelLandmarks) {
					try {
						string lmIdInDb = d.landmarkMappings.at(ml);
						shared_ptr<Landmark> lm = d.labeledImageSource->getLandmarks().getLandmark(lmIdInDb);
						landmarks.at<float>(0, currentLandmark) = lm->getX();
						landmarks.at<float>(0, currentLandmark + modelLandmarks.size()) = lm->getY();
					}
					catch (std::out_of_range& e) {
						appLogger.error(e.what()); // mapping failed
						return EXIT_FAILURE;
					}
					catch (std::invalid_argument& e) {
						appLogger.error(e.what()); // lm not in db
						return EXIT_FAILURE;
					}
					++currentLandmark;
				}
				trainingData.push_back(std::make_tuple(img, detectedFaces[0], landmarks));
			}
		}

	}



	
	// 2. Calculate the mean-shape of all training images
	//    Afterwards: We could do some procrustes and align all shapes to the calculated mean-shape. But actually just the mean calculated above is a good approximation.
	//    Q: At least do some centering/scaling?
	int numImages = trainingData.size();
	int numModelLandmarks = modelLandmarks.size();
	Mat groundtruthLandmarks(numImages, 2 * numModelLandmarks, CV_32FC1);
	Mat groundtruthLandmarksNormalizedByUnitFacebox(2 * numModelLandmarks, numImages, CV_32FC1);
	int currentImage = 0;
	if (false /*meanPreAlign == something*/) {
		/* // careful: old, doesn't use new row-format yet
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
		}*/
	}
	else if (meanPreAlign == MeanPreAlign::NONE) {
		// just copy the coords over into groundtruthLandmarks
		for (const auto& data : trainingData) {
			Mat img = std::get<0>(data);
			Mat lms = std::get<2>(data);
			Mat groundtruthLandmarksRow = groundtruthLandmarks.row(currentImage);
			lms.copyTo(groundtruthLandmarksRow);
			currentImage++;
		}
	}
	// Take the mean of every row:
	Mat modelMean;
	//cv::reduce(groundtruthLandmarksNormalizedByUnitFacebox, modelMean, 1, CV_REDUCE_AVG); // reduce to 1 column
	cv::reduce(groundtruthLandmarks, modelMean, 0, CV_REDUCE_AVG); // reduce to 1 row

	if (meanNormalization == MeanNormalization::UNIT_SUM_SQUARED_NORMS) {
		Mat modelMeanX = modelMean.colRange(0, numModelLandmarks);
		Mat modelMeanY = modelMean.colRange(numModelLandmarks, 2 * numModelLandmarks);
		// calculate the centroid:
		cv::Scalar cx = cv::mean(modelMeanX);
		cv::Scalar cy = cv::mean(modelMeanY);
		// move all points to the centroid
		modelMeanX = modelMeanX - cx[0];
		modelMeanY = modelMeanY - cy[0];
		// scale so that the average norm is 1/numLandmarks (i.e.: the total norm of all vectors added is 1).
		// note: that doesn't make too much sense, because it follows that the more landmarks we use, the smaller the mean-face will be.
		float currentTotalSquaredNorm = 0.0f;
		for (int p = 0; p < numModelLandmarks; ++p) {
			float x = modelMeanX.at<float>(p);
			float y = modelMeanY.at<float>(p);
			currentTotalSquaredNorm += (x*x + y*y);
		}
		// multiply every vectors coordinate by the sqrt of the currentTotalSquaredNorm
		modelMean /= std::sqrt(currentTotalSquaredNorm);
	}
	
	std::ofstream myfile;
	myfile.open("example.txt");
	myfile << "x = [";
	for (int i = 0; i < numModelLandmarks; ++i) {
		myfile << modelMean.at<float>(0, i) << ", ";
	}
	myfile << "];" << std::endl << "y = [";
	for (int i = 0; i < numModelLandmarks; ++i) {
		myfile << modelMean.at<float>(0, i + numModelLandmarks) << ", ";
	}
	myfile << "];" << std::endl;
	myfile.close();
	// Q: Why is our mean twice as big? Their mean doesn't fit into a V&J facebox from -0.5 to 0.5?


	// 2.9 Calculate the mean and variances of the translational and scaling differences between the initial and true landmark locations. (used for generating the samples)
	// This also includes the scaling/translation necessary to go from the unit-sqnorm normalized mean to one in a reasonably sized one w.r.t. the face-box.
	// This means we have to divide the stddev we draw by 2. The translation is ok though.
	// Todo: We should directly learn a reasonably normalized mean during the training!
	Mat delta_tx(numImages, 1, CV_32FC1);
	Mat delta_ty(numImages, 1, CV_32FC1);
	Mat delta_sx(numImages, 1, CV_32FC1);
	Mat delta_sy(numImages, 1, CV_32FC1);
	Mat initialShape = Mat::zeros((numSamplesPerImage+1) * trainingData.size(), 2 * numModelLandmarks, CV_32FC1); // 10 samples + the original data = 11
	currentImage = 0;
	for (const auto& data : trainingData) {
		Mat img = std::get<0>(data);
		cv::Rect detectedFace = std::get<1>(data); // Caution: Depending on flags selected earlier, we might not have detected faces yet!
		
		// calculate the centroid and the min-max bounding-box (for the width/height) of the ground-truth:
		Scalar gtMeanX = cv::mean(groundtruthLandmarks.row(currentImage).colRange(0, numModelLandmarks));
		Scalar gtMeanY = cv::mean(groundtruthLandmarks.row(currentImage).colRange(numModelLandmarks, numModelLandmarks * 2));
		double minWidth, maxWidth, minHeight, maxHeight;
		cv::minMaxIdx(groundtruthLandmarks.row(currentImage).colRange(0, numModelLandmarks), &minWidth, &maxWidth);
		cv::minMaxIdx(groundtruthLandmarks.row(currentImage).colRange(numModelLandmarks, numModelLandmarks * 2), &minHeight, &maxHeight);
		//cv::rectangle(img, cv::Rect(minWidth, minHeight, maxWidth - minWidth, maxHeight - minHeight), Scalar(255.0f, 0.0f, 0.0f));
		
		for (int i = 0; i < numModelLandmarks; ++i) {
			//cv::circle(img, Point2f(groundtruthLandmarks.at<float>(currentImage, i), groundtruthLandmarks.at<float>(currentImage, i + numModelLandmarks)), 3, Scalar(0.0f, 255.0f, 0.0f));
		}

		// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
		// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
		// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
		Mat initialShapeEstimateX0 = modelMean.clone();
		Mat initialShapeEstimateX0_x = initialShapeEstimateX0.colRange(0, initialShapeEstimateX0.cols / 2);
		Mat initialShapeEstimateX0_y = initialShapeEstimateX0.colRange(initialShapeEstimateX0.cols / 2, initialShapeEstimateX0.cols);
		initialShapeEstimateX0_x = (initialShapeEstimateX0_x + 0.5f) * detectedFace.width + detectedFace.x;
		initialShapeEstimateX0_y = (initialShapeEstimateX0_y + 0.5f) * detectedFace.height + detectedFace.y;
		Mat initialShapeRow = initialShape.row(currentImage * (numSamplesPerImage + 1));
		initialShapeEstimateX0.copyTo(initialShapeRow);
		for (int i = 0; i < numModelLandmarks; ++i) {
			//cv::circle(img, Point2f(initialShapeEstimateX0.at<float>(0, i), initialShapeEstimateX0.at<float>(0, i + numModelLandmarks)), 3, Scalar(255.0f, 0.0f, 0.0f));
		}
		//cv::rectangle(img, detectedFace, Scalar(0.0f, 0.0f, 255.0f));

		// calculate the centroid and the min-max bounding-box (for the width/height) of the initial estimate x_0:
		Scalar x0MeanX = cv::mean(initialShapeEstimateX0_x);
		Scalar x0MeanY = cv::mean(initialShapeEstimateX0_y);
		double minWidthX0, maxWidthX0, minHeightX0, maxHeightX0;
		cv::minMaxIdx(initialShapeEstimateX0_x, &minWidthX0, &maxWidthX0);
		cv::minMaxIdx(initialShapeEstimateX0_y, &minHeightX0, &maxHeightX0);
		//cv::rectangle(img, cv::Rect(minWidthX0, minHeightX0, maxWidthX0 - minWidthX0, maxHeightX0 - minHeightX0), Scalar(255.0f, 0.0f, 0.0f));

		//cv::circle(img, Point2f(gtMeanX[0], gtMeanY[0]), 2, Scalar(0.0f, 0.0f, 255.0f)); // gt
		//cv::circle(img, Point2f(x0MeanX[0], x0MeanY[0]), 2, Scalar(0.0f, 255.0f, 255.0f)); // x0
		
		delta_tx.at<float>(currentImage) = (gtMeanX[0] - x0MeanX[0]) / detectedFace.width; // This is in relation to the V&J face-box
		delta_ty.at<float>(currentImage) = (gtMeanY[0] - x0MeanY[0]) / detectedFace.height;
		delta_sx.at<float>(currentImage) = (maxWidth - minWidth) / (maxWidthX0 - minWidthX0);
		delta_sy.at<float>(currentImage) = (maxHeight - minHeight) / (maxHeightX0 - minHeightX0);

	/*	Mat initialShapeEstimate2X0 = modelMean.clone();
		Mat initialShapeEstimate2X0_x = initialShapeEstimate2X0.colRange(0, initialShapeEstimate2X0.cols / 2);
		Mat initialShapeEstimate2X0_y = initialShapeEstimate2X0.colRange(initialShapeEstimate2X0.cols / 2, initialShapeEstimate2X0.cols);
		initialShapeEstimate2X0_x = (initialShapeEstimate2X0_x * delta_sx + 0.5f) * detectedFace.width + detectedFace.x + delta_tx;
		initialShapeEstimate2X0_y = (initialShapeEstimate2X0_y * delta_sy + 0.5f) * detectedFace.height + detectedFace.y + delta_ty;
		Mat initialShapeRow2 = initialShape.row(currentImage * (numSamplesPerImage + 1));
		initialShapeEstimate2X0.copyTo(initialShapeRow);
		for (int i = 0; i < numModelLandmarks; ++i) {
			cv::circle(img, Point2f(initialShapeEstimate2X0.at<float>(0, i), initialShapeEstimate2X0.at<float>(0, i + numModelLandmarks)), 3, Scalar(255.0f, 0.0f, 0.0f));
		}*/


		// Initial esti: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
		/*Mat initialShapeEstimateX0 = modelMean.clone();
		Mat initialShapeEstimateX0_x = initialShapeEstimateX0.rowRange(0, initialShapeEstimateX0.rows / 2);
		Mat initialShapeEstimateX0_y = initialShapeEstimateX0.rowRange(initialShapeEstimateX0.rows / 2, initialShapeEstimateX0.rows);
		initialShapeEstimateX0_x = (initialShapeEstimateX0_x + 0.5f) * detectedFace.width + detectedFace.x;
		initialShapeEstimateX0_y = (initialShapeEstimateX0_y + 0.5f) * detectedFace.height + detectedFace.y;
		Mat initialShapeColumn = initialShape.col(currentImage * (numSamplesPerImage+1));
		initialShapeEstimateX0.copyTo(initialShapeColumn);
		for (int i = 0; i < numModelLandmarks; ++i) {
			cv::circle(img, Point2f(initialShapeEstimateX0.at<float>(i, 0), initialShapeEstimateX0.at<float>(i + numModelLandmarks, 0)), 3, Scalar(0.0f, 255.0f, 0.0f));
		}*/

		// Calculate the mean and variances of the translational and scaling differences between the initial and true landmark locations:
		// Q: how?
		currentImage++;
	}
	// Calculate the mean/variances and store them in doubles:
	double mu_t_x, mu_t_y, mu_s_x, mu_s_y, sigma_t_x, sigma_t_y, sigma_s_x, sigma_s_y;
	{
		Mat mmu_t_x, mmu_t_y, mmu_s_x, mmu_s_y, msigma_t_x, msigma_t_y, msigma_s_x, msigma_s_y;
		cv::meanStdDev(delta_tx, mmu_t_x, msigma_t_x);
		cv::meanStdDev(delta_ty, mmu_t_y, msigma_t_y);
		cv::meanStdDev(delta_sx, mmu_s_x, msigma_s_x);
		cv::meanStdDev(delta_sy, mmu_s_y, msigma_s_y);
		mu_t_x = mmu_t_x.at<double>(0);
		mu_t_y = mmu_t_y.at<double>(0);
		mu_s_x = mmu_s_x.at<double>(0);
		mu_s_y = mmu_s_y.at<double>(0);
		sigma_t_x = msigma_t_x.at<double>(0);
		sigma_t_y = msigma_t_y.at<double>(0);
		sigma_s_x = msigma_s_x.at<double>(0);
		sigma_s_y = msigma_s_y.at<double>(0);
	}
	// Rescale the model-mean, and the mean variances as well: (only necessary if our mean is not normalized to V&J face-box directly in first steps)
	Mat modelMean_x = modelMean.colRange(0, modelMean.cols / 2);
	Mat modelMean_y = modelMean.colRange(modelMean.cols / 2, modelMean.cols);
	modelMean_x = (modelMean_x * mu_s_x) + mu_t_x;
	modelMean_y = (modelMean_y * mu_s_y) + mu_t_y;
	sigma_s_x = sigma_s_x / mu_s_x;
	mu_s_x = 1.0;
	sigma_s_y = sigma_s_y / mu_s_y;
	mu_s_y = 1.0;
	mu_t_x = 0;
	mu_t_y = 0;

	myfile.open("example.txt");
	myfile << "x = [";
	for (int i = 0; i < numModelLandmarks; ++i) {
		myfile << modelMean.at<float>(0, i) << ", ";
	}
	myfile << "];" << std::endl << "y = [";
	for (int i = 0; i < numModelLandmarks; ++i) {
		myfile << modelMean.at<float>(0, i + numModelLandmarks) << ", ";
	}
	myfile << "];" << std::endl;
	myfile.close();

	// 3. For every training image:
	// Store the initial shape estimate (x_0) of the image, plus generate 10 samples and store them as well
	std::mt19937 engine; ///< A Mersenne twister MT19937 engine
	engine.seed();
	std::normal_distribution<float> rndN_t_x(mu_t_x, sigma_t_x);
	std::normal_distribution<float> rndN_t_y(mu_t_y, sigma_t_y);
	std::normal_distribution<float> rndN_s_x(mu_s_x, sigma_s_x);
	std::normal_distribution<float> rndN_s_y(mu_s_y, sigma_s_y);
	currentImage = 0;
	for (const auto& data : trainingData) {
		// a) Run the face-detector (the same that we're going to use in the testing-stage) (already done)
		Mat img = std::get<0>(data);
		cv::Rect detectedFace = std::get<1>(data); // TODO: Careful, depending on options, we might not have face-boxes yet
		// b) Align the model to the current face-box. (rigid, only centering of the mean). x_0
		// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
		// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
		// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
		Mat initialShapeEstimateX0 = modelMean.clone();
		Mat initialShapeEstimateX0_x = initialShapeEstimateX0.colRange(0, initialShapeEstimateX0.cols / 2);
		Mat initialShapeEstimateX0_y = initialShapeEstimateX0.colRange(initialShapeEstimateX0.cols / 2, initialShapeEstimateX0.cols);
		initialShapeEstimateX0_x = (initialShapeEstimateX0_x + 0.5f) * detectedFace.width + detectedFace.x;
		initialShapeEstimateX0_y = (initialShapeEstimateX0_y + 0.5f) * detectedFace.height + detectedFace.y;
		Mat initialShapeRow = initialShape.row(currentImage * (numSamplesPerImage + 1));
		initialShapeEstimateX0.copyTo(initialShapeRow);
		for (int i = 0; i < numModelLandmarks; ++i) {
			//cv::circle(img, Point2f(initialShapeEstimateX0.at<float>(0, i), initialShapeEstimateX0.at<float>(0, i + numModelLandmarks)), 2, Scalar(255.0f, 0.0f, 0.0f));
		}
		//cv::rectangle(img, detectedFace, Scalar(0.0f, 0.0f, 255.0f));
		// c) Generate Monte Carlo samples? With what variance? x_0^i (maybe make this step 3.)
		// sample around initialShapeThis, store in initialShape
		//		Save the samples, all in a matrix
		//		Todo 1) don't use pixel variance, but a scale-independent one (normalize by IED?)
		//			 2) calculate the variance from data (gt facebox?)
		for (int sample = 0; sample < numSamplesPerImage; ++sample) {
			Mat initialShapeEstimateX0 = initialShape.row(currentImage*(numSamplesPerImage + 1) + (sample + 1));
			Mat initialShapeEstimateX0_x = initialShapeEstimateX0.colRange(0, initialShapeEstimateX0.cols / 2);
			Mat initialShapeEstimateX0_y = initialShapeEstimateX0.colRange(initialShapeEstimateX0.cols / 2, initialShapeEstimateX0.cols);
			modelMean.copyTo(initialShapeEstimateX0);
			initialShapeEstimateX0_x = ((initialShapeEstimateX0_x)*rndN_s_x(engine) + 0.5f + rndN_t_x(engine)) * detectedFace.width + detectedFace.x;
			initialShapeEstimateX0_y = ((initialShapeEstimateX0_y)*rndN_s_y(engine) + 0.5f + rndN_t_y(engine)) * detectedFace.height + detectedFace.y;
			// Check if the sample goes outside the feature-extractable region?
			// TODO: The scaling needs to be done in the normalized facebox region?? Try to write it down?
			// Better do the translation in the norm-FB as well to be independent of face-size? yes we do that now. Check the Detection-code though!
			for (int i = 0; i < numModelLandmarks; ++i) {
				//cv::circle(img, Point2f(initialShapeEstimateX0.at<float>(0, i), initialShapeEstimateX0.at<float>(0, i + numModelLandmarks)), 2, Scalar(0.0f, 0.0f, 255.0f));
			}
		}
		currentImage++;
	}
	// 4. For every sample plus the original image: (loop through the matrix)
	//			a) groundtruthShape, initialShape
	//				deltaShape = ground - initial
	// Duplicate each row in groundtruthLandmarks for every sample, store in groundtruthShapes
	Mat groundtruthShapes = Mat::zeros((numSamplesPerImage+1) * trainingData.size(), 2 * numModelLandmarks, CV_32FC1); // 10 samples + the original data = 11
	for (int currImg = 0; currImg < groundtruthLandmarks.rows; ++currImg) {
		Mat groundtruthLandmarksRow = groundtruthLandmarks.row(currImg);
		for (int j = 0; j < numSamplesPerImage + 1; ++j) {
			Mat groundtruthShapesRow = groundtruthShapes.row(currImg*(numSamplesPerImage + 1) + j);
			groundtruthLandmarksRow.copyTo(groundtruthShapesRow);
		}
	}
	Mat deltaShape = groundtruthShapes - initialShape;
	//			b) Extract the features at all landmark locations initialShape (Paper: SIFT, 32x32 (?))
	vector<string> descriptorTypes;
	vector<shared_ptr<FeatureDescriptorExtractor>> descriptorExtractors;
	shared_ptr<FeatureDescriptorExtractor> sift = make_shared<SiftFeatureDescriptorExtractor>();
	// read params if there are any?
	descriptorExtractors.push_back(sift);
	descriptorTypes.push_back("OpenCVSift");
	//int featureDimension = 128;
	Mat featureMatrix;// = Mat::ones(initialShape.rows, (featureDimension * numModelLandmarks) + 1, CV_32FC1); // Our 'A'. The last column stays all 1's; it's for learning the offset/bias
	currentImage = 0;
	for (const auto& data : trainingData) {
		Mat img = std::get<0>(data);
		for (int sample = 0; sample < numSamplesPerImage + 1; ++sample) {
			vector<cv::Point2f> keypoints;
			for (int lm = 0; lm < numModelLandmarks; ++lm) {
				float px = initialShape.at<float>(currentImage*(numSamplesPerImage + 1) + sample, lm);
				float py = initialShape.at<float>(currentImage*(numSamplesPerImage + 1) + sample, lm + numModelLandmarks);
				keypoints.emplace_back(cv::Point2f(px, py));
			}
			Mat featureDescriptors = sift->getDescriptors(img, keypoints);
			// concatenate all the descriptors for this sample horizontally (into a row-vector)
			featureDescriptors = featureDescriptors.reshape(0, featureDescriptors.cols * numModelLandmarks).t();
			//int currentImageMatrixIndex = currentImage*(numSamplesPerImage + 1) + sample;
			//			c) store the features
			//Mat featureMatrixRow = featureMatrix.row(currentImageMatrixIndex).colRange(0, featureDimension * numModelLandmarks); // take everything up to the last column (which is a 1)
			//featureDescriptors.copyTo(featureMatrixRow);
			featureMatrix.push_back(featureDescriptors);
		}
		++currentImage;
	}
	Mat biasColumn = Mat::ones(initialShape.rows, 1, CV_32FC1);
	cv::hconcat(featureMatrix, biasColumn, featureMatrix); // Other options: 1) Generate one bigger Mat and use copyTo (memory would be continuous then) or 2) implement a FeatureDescriptorExtractor::getDimension()
	
	// 5. Add one row to the features (already done), add regLambda
	Mat AtA = featureMatrix.t() * featureMatrix;
	float lambda = cv::norm(AtA);
	lambda = 0.5f * lambda / (numImages*(numSamplesPerImage + 1));
	Mat regulariser = Mat::eye(AtA.rows, AtA.rows, CV_32FC1) * lambda;
	regulariser.at<float>(regulariser.rows - 1, regulariser.cols - 1) = 0.0f; // no lambda for the bias
	//		solve for x!
	Mat AtAReg = AtA + regulariser;
	Mat AtARegInv = AtAReg.inv(cv::DECOMP_SVD); // default is LU
	Mat AtARegInvAt = AtARegInv * featureMatrix.t();
	Mat AtARegInvAtb = AtARegInvAt * deltaShape; // = x

	Mat R = AtARegInvAtb;

	std::vector<cv::Mat> regressorData;
	regressorData.push_back(R);

	// Do the following:
	// * Save the model, i.e.: Write everything to a SDM Model, then sdmmodel::Save
	// * TEST IT! tracker/fitting/txt/ML script etc anpassen, neues row-format, params etc
	// * move everything to a SupervisedDescentTraining class, i.e. LandmarkBasedSupervisedDescentTraining (later: GenericSupervisedDescentTraining? abstract base-class or not?)
	// - move below code up, dont duplicate code
	// - after training R, evaluate, print error, save the new shapeMatrix
	// - Then loop! (which means the code above starts after the shape-matrix (and everything else) has been prepared
	// - time measurement
	// - calculate the error after each regressor is learned.
	// - update ML script for ZF's model
	// - more params, mean etc, my mean...
	// - draw curves
	// - impl numcascade steps config

	// 6. Do the same again for all cascade steps
	int numCascadeMaxSteps = 3;
	//float cascadeErrorThreshold = 2.0f;

	int currentCascadeStep = 1;
	//float currentError = 5.0f;
	while (currentCascadeStep < numCascadeMaxSteps/* && currentError > cascadeErrorThreshold*/) {
		shared_ptr<FeatureDescriptorExtractor> sift = make_shared<SiftFeatureDescriptorExtractor>();
		// read params if there are any? (generate according to config)
		descriptorExtractors.push_back(sift);
		descriptorTypes.push_back("OpenCVSift");
		Mat featureMatrixThisStep;
		for (int currentImage = 0; currentImage < trainingData.size(); ++currentImage) {
			Mat img = std::get<0>(trainingData[currentImage]);
			Mat output = img.clone();
			for (int sample = 0; sample < numSamplesPerImage + 1; ++sample) {
				int currentRowInAllData = currentImage * (numSamplesPerImage + 1) + sample;
				// gt:
				for (int i = 0; i < numModelLandmarks; ++i) {
					cv::circle(output, Point2f(groundtruthShapes.at<float>(currentRowInAllData, i), groundtruthShapes.at<float>(currentRowInAllData, i + numModelLandmarks)), 2, Scalar(255.0f, 0.0f, 0.0f));
				}
				// x0:
				for (int i = 0; i < numModelLandmarks; ++i) {
					cv::circle(output, Point2f(initialShape.at<float>(currentRowInAllData, i), initialShape.at<float>(currentRowInAllData, i + numModelLandmarks)), 2, Scalar(210.0f, 255.0f, 0.0f));
				}
				Mat shapeStep = featureMatrix.row(currentRowInAllData) * R;
				Mat x1 = initialShape.row(currentRowInAllData).t() + shapeStep.t(); // add columns
				x1 = x1.t(); // now a row-vector
				Mat initialShapeRow = initialShape.row(currentRowInAllData);
				x1.copyTo(initialShapeRow);
				// x1:
				for (int i = 0; i < numModelLandmarks; ++i) {
					cv::circle(output, Point2f(x1.at<float>(i), x1.at<float>(i + numModelLandmarks)), 2, Scalar(255.0f, 185.0f, 0.0f));
				}

				// Feature extr.: Very similar code to above step0
				vector<cv::Point2f> keypoints;
				for (int lm = 0; lm < numModelLandmarks; ++lm) {
					float px = initialShape.at<float>(currentImage*(numSamplesPerImage + 1) + sample, lm);
					float py = initialShape.at<float>(currentImage*(numSamplesPerImage + 1) + sample, lm + numModelLandmarks);
					keypoints.emplace_back(cv::Point2f(px, py));
				}
				Mat featureDescriptors = sift->getDescriptors(img, keypoints);
				// concatenate all the descriptors for this sample horizontally (into a row-vector)
				featureDescriptors = featureDescriptors.reshape(0, featureDescriptors.cols * numModelLandmarks).t();
				//int currentImageMatrixIndex = currentImage*(numSamplesPerImage + 1) + sample;
				//			c) store the features
				//Mat featureMatrixRow = featureMatrix.row(currentImageMatrixIndex).colRange(0, featureDimension * numModelLandmarks); // take everything up to the last column (which is a 1)
				//featureDescriptors.copyTo(featureMatrixRow);
				featureMatrixThisStep.push_back(featureDescriptors);
			}
		}
		featureMatrix = featureMatrixThisStep;
		Mat biasColumn = Mat::ones(initialShape.rows, 1, CV_32FC1);
		cv::hconcat(featureMatrix, biasColumn, featureMatrix); // Other options: 1) Generate one bigger Mat and use copyTo (memory would be continuous then) or 2) implement a FeatureDescriptorExtractor::getDimension()

		Mat deltaShape = groundtruthShapes - initialShape;

		Mat AtA = featureMatrix.t() * featureMatrix;
		float lambda = 0.5f * cv::norm(AtA) / (numImages*(numSamplesPerImage + 1));;
		Mat regulariser = Mat::eye(AtA.rows, AtA.rows, CV_32FC1) * lambda;
		regulariser.at<float>(regulariser.rows - 1, regulariser.cols - 1) = 0.0f; // no lambda for the bias
		Mat AtAReg = AtA + regulariser;
		Mat AtARegInvAtb = AtAReg.inv(cv::DECOMP_SVD) * featureMatrix.t() * deltaShape;

		R = AtARegInvAtb;
		regressorData.push_back(R);

		//cascadeErrorThreshold = 4.0f;
		++currentCascadeStep;
	}

	std::time_t currentTime_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	string currentTime = std::ctime(&currentTime_t);

	SdmLandmarkModel model(modelMean, modelLandmarks, regressorData, descriptorExtractors, descriptorTypes);
	model.save(outputFilename, "Trained on " + currentTime.substr(0, currentTime.length() - 1));
	appLogger.info("Finished training. Saved model to " + outputFilename.string() + ".");

	return 0;
}
