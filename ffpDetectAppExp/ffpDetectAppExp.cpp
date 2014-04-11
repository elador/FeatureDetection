/*
 * ffpDetectAppExp.cpp
 *
 *  Created on: 22.03.2013
 *      Author: Patrik Huber
 */

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
#define _CRTDBG_MAP_ALLOC
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

#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <numeric>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"

#include "classification/RbfKernel.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/WvmClassifier.hpp"
#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/ProbabilisticRvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/RectLandmark.hpp"
#include "imageio/ModelLandmark.hpp"
#include "imageio/IbugLandmarkFormatParser.hpp"
#include "imageio/LstLandmarkFormatParser.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/NamedLabeledImageSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/FddbLandmarkSink.hpp"

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/ReshapingFilter.hpp"
#include "imageprocessing/ConversionFilter.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/FilteringPyramidFeatureExtractor.hpp"
#include "imageprocessing/FilteringFeatureExtractor.hpp"
#include "imageprocessing/HistEq64Filter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"
#include "imageprocessing/ZeroMeanUnitVarianceFilter.hpp"
#include "imageprocessing/UnitNormFilter.hpp"
#include "imageprocessing/WhiteningFilter.hpp"

#include "detection/SlidingWindowDetector.hpp"
#include "detection/ClassifiedPatch.hpp"
#include "detection/OverlapElimination.hpp"
#include "detection/FiveStageSlidingWindowDetector.hpp"

#include "shapemodels/MorphableModel.hpp"
#include "shapemodels/FeaturePointsModelRANSACtmp.hpp"
#include "shapemodels/RansacFeaturePointsModel.hpp"
#include "shapemodels/FeaturePointsSelector.hpp"
#include "shapemodels/FeaturePointsEvaluator.hpp"

#include "logging/LoggerFactory.hpp"
#include "imagelogging/ImageLoggerFactory.hpp"
#include "imagelogging/ImageFileWriter.hpp"

namespace po = boost::program_options;
using namespace std;
using namespace imageprocessing;
using namespace detection;
using namespace classification;
using namespace imageio;
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;
using imagelogging::ImageLogger;
using imagelogging::ImageLoggerFactory;
using boost::property_tree::ptree;
using boost::property_tree::info_parser::read_info;
using boost::filesystem::path;
using boost::lexical_cast;


void drawBoxes(Mat image, vector<shared_ptr<ClassifiedPatch>> patches)
{
	for(const auto& cpatch : patches) {
		shared_ptr<Patch> patch = cpatch->getPatch();
		cv::rectangle(image, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((cpatch->getProbability())/1.0)   ));
	}
}


void drawFfpsCircle(Mat image, pair<string, Point2f> landmarks)
{
	cv::Point center(cvRound(landmarks.second.x), cvRound(landmarks.second.y));
	int radius = cvRound(3);
	circle(image, center, 1, cv::Scalar(0,255,0), 1, 8, 0 );	// draw the circle center
	circle(image, center, radius, cv::Scalar(0,0,255), 1, 8, 0 );	// draw the circle outline

}

void drawFfpsText(Mat image, pair<string, Point2f> landmarks)
{
	cv::Point center(cvRound(landmarks.second.x), cvRound(landmarks.second.y));
	std::ostringstream text;
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 0.7;
	int thickness = 1;  
	text << landmarks.first << std::ends;
	putText(image, text.str(), center, fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
	text.str("");
}

/**
 * Takes a list of classified patches and creates a single probability map of face region locations.
 *
 * Note/TODO: We could increase the probability of a region when nearby patches (in x/y and scale)
 *            also say it's a face. But that might be very difficult with this current approach.
 *            27.09.2013 16:49
 * 
 * @param[in] width The width of the original image where the classifier was run.
 * @param[in] height The height of the original image where the classifier was run.
 * @return A probability map for face regions with float values between 0 and 1.
 */
Mat getFaceRegionProbabilityMapFromPatchlist(vector<shared_ptr<ClassifiedPatch>> patches, int width, int height)
{
	Mat faceRegionProbabilityMap(height, width, CV_32FC1, cv::Scalar(0.0f));
	
	for (auto patch : patches) {
		const unsigned int pw = patch->getPatch()->getBounds().width;
		const unsigned int ph = patch->getPatch()->getBounds().height;
		const unsigned int px = patch->getPatch()->getBounds().x;
		const unsigned int py = patch->getPatch()->getBounds().y;
		for (unsigned int currX = px; currX < px+pw-1; ++currX) {	// Note: I'm not exactly sure why the "-1" is necessary,
			for (unsigned int currY = py; currY < py+ph-1; ++currY) { // but without it, it goes beyond the image bounds

				if(currX>=faceRegionProbabilityMap.cols) {
					cv::imwrite("TEST.png", patch->getPatch()->getData());
				}

				if(currX < faceRegionProbabilityMap.cols && currY < faceRegionProbabilityMap.rows) { // Note: This is a temporary check, as long as we
					if (patch->getProbability() > faceRegionProbabilityMap.at<float>(currY, currX)) { // haven't fixed that up/downscaling rounding problem
						faceRegionProbabilityMap.at<float>(currY, currX) = patch->getProbability();   // that patches can be outside the original image.
					}
				}
			}
		}
	}
	/* Idea for improvement:
		Create a probability map for each scale first (the centers, not the region).
		Then, weight each point with the surrounding 8 (or more, or also in scale-dir)
		detections. This is a re-weighting of the probabilities. Then calculate the new
		face-region-probMap. (or do/combine this directly?)
	*/

	return faceRegionProbabilityMap;
}

Mat patchToMask(shared_ptr<const Patch> patch, Mat mask)
{
	//cv::rectangle(mask, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((cpatch->getProbability())/1.0)   ));
	unsigned int ty = patch->getY() - patch->getHeight()/2.0f;
	unsigned int by = patch->getY() + patch->getHeight()/2.0f;
	unsigned int lx = patch->getX() - patch->getWidth()/2.0f;
	unsigned int rx = patch->getX() + patch->getWidth()/2.0f;
	for (unsigned int y = ty; y < by; ++y) {
		for (unsigned int x = lx; x < rx; ++x) { // Note: Might suffer from the same out-of-range problem than getFaceRegionProbabilityMapFromPatchlist(...).
			mask.at<uchar>(y, x) = 1;
		}
	}
	return mask;
}

void doNothing() {};

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
	return os;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	string verboseLevelImages;
	bool useFileList = false;
	bool useImgs = false;
	bool useDirectory = false;
	bool useGroundtruth = false;
	vector<path> inputPaths;
	path inputFilelist;
	path inputDirectory;
	vector<path> inputFilenames;
	path configFilename;
	shared_ptr<ImageSource> imageSource;
	path outputPicsDir; // TODO: ImageLogger vs ImageSinks? (see AdaptiveTracking.cpp)
	path groundtruthDir; // TODO: Make more dynamic wrt landmark format. a) What about the loading-flags (1_Per_Folder etc) we have? b) Expose those flags to cmdline? c) Make a LmSourceLoader and he knows about a LM_TYPE (each corresponds to a Parser/Loader class?)
	// TODO Also, sometimes we might have the face-box annotated but not LMs, sometimes only LMs and no Facebox.
	string groundtruthType;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("verbose-images,w", po::value<string>(&verboseLevelImages)->implicit_value("INTERMEDIATE")->default_value("FINAL","write images with FINAL loglevel or below."),
				  "specify the verbosity of the image output: FINAL, INTERMEDIATE, INFO, DEBUG or TRACE")
			("config,c", po::value<path>(&configFilename)->required(), 
				"path to a config (.cfg) file")
			("input,i", po::value<vector<path>>(&inputPaths)->required(), 
				"input from one or more files, a directory, or a  .lst/.txt-file containing a list of images")
			("groundtruth,g", po::value<path>(&groundtruthDir), 
				"load ground truth landmarks from the given folder along with the images and output statistics of the detection results")
			("groundtruth-type,t", po::value<string>(&groundtruthType), 
				"specify the type of landmarks to load: lst, ibug")
			("output-dir,o", po::value<path>(&outputPicsDir)->default_value("."),
				"output directory for the result images")
		;

		po::positional_options_description p;
		p.add("input", -1);
		
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << "Usage: ffpDetectApp [options]\n";
			cout << desc;
			return EXIT_SUCCESS;
		}
	
		if (vm.count("groundtruth")) {
			useGroundtruth = true;
			if (!vm.count("groundtruth-type")) {
				cout << "You have specified to use ground truth. Please also specify the type of the landmarks to load via --groundtruth-type or -t." << endl;
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
	imagelogging::loglevel imageLogLevel;
	if(boost::iequals(verboseLevelImages, "FINAL")) imageLogLevel = imagelogging::loglevel::FINAL;
	else if(boost::iequals(verboseLevelImages, "INTERMEDIATE")) imageLogLevel = imagelogging::loglevel::INTERMEDIATE;
	else if(boost::iequals(verboseLevelImages, "INFO")) imageLogLevel = imagelogging::loglevel::INFO;
	else if(boost::iequals(verboseLevelImages, "DEBUG")) imageLogLevel = imagelogging::loglevel::DEBUG;
	else if(boost::iequals(verboseLevelImages, "TRACE")) imageLogLevel = imagelogging::loglevel::TRACE;
	else {
		cout << "Error: Invalid image loglevel." << endl;
		return EXIT_FAILURE;
	}
	
	Loggers->getLogger("classification").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("imageprocessing").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("detection").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("shapemodels").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("ffpDetectApp").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("ffpDetectApp");

	appLogger.debug("Verbose level for console output: " + logging::loglevelToString(logLevel));
	appLogger.debug("Verbose level for image output: " + imagelogging::loglevelToString(imageLogLevel));
	appLogger.debug("Using config: " + configFilename.string());
	appLogger.debug("Using output directory: " + outputPicsDir.string());
	if(outputPicsDir == ".") {
		appLogger.info("Writing output images into current directory.");
	}

	ImageLoggers->getLogger("detection").addAppender(make_shared<imagelogging::ImageFileWriter>(imageLogLevel, outputPicsDir));
	ImageLoggers->getLogger("app").addAppender(make_shared<imagelogging::ImageFileWriter>(imageLogLevel, outputPicsDir / "final"));

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
	if (useGroundtruth) {
		vector<path> groundtruthDirs; groundtruthDirs.push_back(groundtruthDir); // Todo: Make cmdline use a vector<path>
		shared_ptr<LandmarkFormatParser> landmarkFormatParser;
		if(boost::iequals(groundtruthType, "lst")) {
			landmarkFormatParser = make_shared<LstLandmarkFormatParser>();
			landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, string(), GatherMethod::SEPARATE_FILES, groundtruthDirs), landmarkFormatParser);
		} else if(boost::iequals(groundtruthType, "ibug")) {
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
		read_info(configFilename.string(), pt);
	} catch(const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	unordered_map<string, shared_ptr<Detector>> faceDetectors;
	unordered_map<string, shared_ptr<Detector>> featureDetectors;

	try {
		ptree ptDetectors = pt.get_child("detectors");
		for (const auto& kv : ptDetectors) { // kv is of type ptree::value_type

			string landmarkName = kv.second.get<string>("landmark");
			string type = kv.second.get<string>("type");

			if(type=="fiveStageCascade") {

				ptree firstClassifierNode = kv.second.get_child("firstClassifier");
				ptree secondClassifierNode = kv.second.get_child("secondClassifier");
				ptree imgpyr = kv.second.get_child("pyramid");
				ptree oeCfg = kv.second.get_child("overlapElimination");

				shared_ptr<ProbabilisticWvmClassifier> firstClassifier = ProbabilisticWvmClassifier::load(firstClassifierNode); // TODO: Lots of todos here with numUsedFilters, bias, ...
				shared_ptr<ProbabilisticSvmClassifier> secondClassifier = ProbabilisticSvmClassifier::load(secondClassifierNode);

				shared_ptr<OverlapElimination> oe = make_shared<OverlapElimination>(oeCfg.get<float>("dist", 5.0f), oeCfg.get<float>("ratio", 0.0f));

				shared_ptr<DirectPyramidFeatureExtractor> featureExtractor;
				if (imgpyr.get<int>("patch.minWidth", 0) != 0 && imgpyr.get<int>("patch.maxWidth", 0) != 0) {
					// The user has set values for patch.minWidth and maxWidth - use them:
					featureExtractor = make_shared<DirectPyramidFeatureExtractor>(imgpyr.get<int>("patch.width"), imgpyr.get<int>("patch.height"), imgpyr.get<int>("patch.minWidth"), imgpyr.get<int>("patch.maxWidth"), imgpyr.get<double>("incrementalScaleFactor"));
					featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
				} else {
					// The user didn't set values for patch.width and height, use the scale factors or default values:
					shared_ptr<ImagePyramid> imgPyr = make_shared<ImagePyramid>(imgpyr.get<float>("incrementalScaleFactor", 0.9f), imgpyr.get<float>("minScaleFactor", 0.09f), imgpyr.get<float>("maxScaleFactor", 0.25f));
					imgPyr->addImageFilter(make_shared<GrayscaleFilter>());
					featureExtractor = make_shared<DirectPyramidFeatureExtractor>(imgPyr, imgpyr.get<int>("patch.width"), imgpyr.get<int>("patch.height"));
				}
				// TODO: Make this read from the config file, see code below in 'single'
				featureExtractor->addPatchFilter(make_shared<HistEq64Filter>());

				shared_ptr<SlidingWindowDetector> det = make_shared<SlidingWindowDetector>(firstClassifier, featureExtractor);

				shared_ptr<FiveStageSlidingWindowDetector> fsd = make_shared<FiveStageSlidingWindowDetector>(det, oe, secondClassifier);
				fsd->landmark = landmarkName;
				if (landmarkName == "face")	{
					faceDetectors.insert(make_pair(kv.first, fsd));
				} else {
					featureDetectors.insert(make_pair(kv.first, fsd));
				}

			} else if(type=="single") {

				ptree classifierNode = kv.second.get_child("classifier");
				ptree imgpyr = kv.second.get_child("pyramid");
				ptree featurespace = kv.second.get_child("feature");

				// One for all classifiers (with same pyramids):
				// This:
				shared_ptr<ImagePyramid> imgPyr = make_shared<ImagePyramid>(imgpyr.get<float>("incrementalScaleFactor", 0.9f), imgpyr.get<float>("minScaleFactor", 0.09f), imgpyr.get<float>("maxScaleFactor", 0.25f));
				imgPyr->addImageFilter(make_shared<GrayscaleFilter>());
				shared_ptr<DirectPyramidFeatureExtractor> patchExtractor = make_shared<DirectPyramidFeatureExtractor>(imgPyr, imgpyr.get<int>("patch.width"), imgpyr.get<int>("patch.height"));
				// Or:
				//shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(config.get<int>("pyramid.patch.width"), config.get<int>("pyramid.patch.height"), config.get<int>("pyramid.patch.minWidth"), config.get<int>("pyramid.patch.maxWidth"), config.get<double>("pyramid.scaleFactor"));
				//featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());

				// One for each classifiers, can make several, that share the same DirectPyramidFeatureExtractor
				// The split in FilteringPyramidFeatureExtractor and DirectPyramidFeatureExtractor is theoretically not necessary here
				// as we only have one classifier. But I guess we need it if we start sharing pyramids across several detectors.
				auto featureExtractor = make_shared<FilteringPyramidFeatureExtractor>(patchExtractor);
				if (featurespace.get_value<string>() == "histeq") {
					//ared_ptr<FilteringFeatureExtractor> featureExtractor = make_shared<FilteringFeatureExtractor>(patchExtractor);
					featureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
				} else if (featurespace.get_value<string>() == "whi") {
					//shared_ptr<FilteringFeatureExtractor> featureExtractor = make_shared<FilteringFeatureExtractor>(patchExtractor);
					featureExtractor->addPatchFilter(make_shared<WhiteningFilter>());
					featureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
					featureExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0/127.5, -1.0));
					featureExtractor->addPatchFilter(make_shared<UnitNormFilter>(cv::NORM_L2));
				} else if (featurespace.get_value<string>() == "hq64") {
					//shared_ptr<FilteringFeatureExtractor> featureExtractor = make_shared<FilteringFeatureExtractor>(patchExtractor);
					featureExtractor->addPatchFilter(make_shared<HistEq64Filter>());
				} else if (featurespace.get_value<string>() == "gray") {
					//shared_ptr<FilteringFeatureExtractor> featureExtractor = make_shared<FilteringFeatureExtractor>(patchExtractor);
					// no patch filter
				}

				ptree patchFilterNodes = kv.second.get_child("patchFilter");
				for (const auto& filterNode : patchFilterNodes) {
					string filterType = filterNode.first;
					if (filterType=="reshapingFilter") {
						int filterArgs = filterNode.second.get_value<int>();
						featureExtractor->addPatchFilter(make_shared<ReshapingFilter>(filterArgs));
					} else if (filterType=="conversionFilter") {
						string filterArgs = filterNode.second.get_value<string>();
						stringstream ss(filterArgs);
						int type;
						ss >> type;
						double scaling;
						ss >> scaling;
						featureExtractor->addPatchFilter(make_shared<ConversionFilter>(type, scaling));
					}
				}

				string classifierType = classifierNode.get_value<string>();
				shared_ptr<ProbabilisticClassifier> classifier;
				if (classifierType == "pwvm") {
					classifier = ProbabilisticWvmClassifier::load(classifierNode);
				} else if (classifierType == "prvm") {
					classifier = ProbabilisticRvmClassifier::load(classifierNode);
				} if (classifierType == "psvm") {
					classifier = ProbabilisticSvmClassifier::load(classifierNode);
				} 
				
				shared_ptr<SlidingWindowDetector> det = make_shared<SlidingWindowDetector>(classifier, featureExtractor);

				det->landmark = landmarkName;
				if (landmarkName == "face")	{
					faceDetectors.insert(make_pair(kv.first, det));
				} else {
					featureDetectors.insert(make_pair(kv.first, det));
				}
			}

		}
	} catch(const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	} catch (const invalid_argument& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	} catch (const runtime_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	} catch (const logic_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	bool enableFeaturePointValidation;
	string morphableModelFile;
	string morphableModelVertexMappingFile;
	int numRansacIterations;
	shapemodels::MorphableModel mm;
	shapemodels::RansacFeaturePointsModel featurePointsModel;
	
	try {
		try {
			ptree ptFeaturePointValidation = pt.get_child("featurePointValidation");
			enableFeaturePointValidation = ptFeaturePointValidation.get<bool>("enabled", false);
			if (enableFeaturePointValidation) {
				//string mode = ptFeaturePointValidation.get<string>("mode", "ransac");
				morphableModelFile = ptFeaturePointValidation.get<string>("morphableModel");
				morphableModelVertexMappingFile = ptFeaturePointValidation.get<string>("morphableModelVertexMapping");
				numRansacIterations = ptFeaturePointValidation.get<int>("numIterations", 50);
				mm = shapemodels::MorphableModel::loadScmModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\SurreyLowResGuosheng\\NON3448\\ShpVtxModelBin_NON3448.scm", "C:\\Users\\Patrik\\Documents\\GitHub\\featurePoints_SurreyScm.txt");
				shapemodels::FeaturePointsSelector sel;
				shapemodels::FeaturePointsEvaluator eva(mm);
				featurePointsModel = shapemodels::RansacFeaturePointsModel(sel, eva);
			}
		} catch (const boost::property_tree::ptree_bad_path& e) {
			enableFeaturePointValidation = false;
		}
	} catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	
	FddbLandmarkSink landmarkSink("annotatedList.txt");
	landmarkSink.open(outputPicsDir.string() + "/final/" + "detectedFaces.txt");
	
	

	// lm-loading
	// output-dir
	// load ffd/ROI
	// relative bilder-pfad aus filelist
	// boost::po behaves strangely with -h and the required arguments (cannot show help without them) ?
	// our libs: add library dependencies (eg to boost) in add_library ?
	// log (text) what is going on. Eg detecting on image... bla... Svm reduced from x to y...
	//       where to put this? as deep as possible? (eg just there where the variable needed (eg filename, 
	//       detector-name is still visible). I think for OE there's already something in it.
	// move drawBoxes(...) somewhere else
	// in the config: firstStage/secondStage: What if they have different feature spaces (or patch-sizes). At the
	//      moment, in 1 FiveStageDet., I believe there cannot be 2 different feature spaces. 
	//      (the second classifier just gets a list of patches - theoretically, he could go extract them again?)
	//      Should we make this all way more dynamic?
	// test what happens when I delete the whole config content and run it. where does it take default values, where errors, etc.
	//
	// Note concerning SlidingWindowDetector and FiveStageSlidingWindowDetector: Theoretically we only need SlidingWindowDetector and
	//      we could give it a Two/FiveStageClassifier instead of just the WVM. This way we would only need one SlidingWindowDetector
	//      for all. But the detector would lose the capability to do smart things like using the second-best face-box etc.?
	//      No, the detector only needs something to classify ONE point and get a result.
	//      But a FiveStageClassifier cannot go into libClassification as it works on more than just 1 feature vector. It is not a classifier for one point.

	/* Note: We could change/write/add something to the config with
	pt.put("detection.svm.threshold", -0.5f);
	If the value already exists, it gets overwritten, if not, it gets created.
	Save it with:
	write_info("C:\\Users\\Patrik\\Documents\\GitHub\\ffpDetectApp.cfg", pt);
	*/

	class imageStatistic 
	{
		public:
		enum class DetectionType {
			TACC, ///< todo
			FACC,
			TREJ,
			FREJ,
			NOGROUNDTRUTH
		};
		/* Face */
		bool haveFaceGroundtruth; // delete this, better in NOGROUNDTRUTH?
		size_t numFaceCandidates;
		DetectionType faceDetectionResult;
		/* Landmarks */
		float interEyeDistance;
		bool haveLandmarkGroundtruth;
		vector<float> landmarkPixelError;
		vector<DetectionType> landmarkDetectionResult;
		vector<string> landmarkNames;
		string getFaceStatisticsString() {
			string faceDetResult;
			if (!haveFaceGroundtruth) {
				faceDetResult = lexical_cast<string>(numFaceCandidates) + " face candidates detected, but no ground truth available.";
			} else {
				faceDetResult = "Face detection result: " + detectionTypeToString(faceDetectionResult);
			}
			return faceDetResult;
		};
		string getLandmarksStatisticsString() {
			float pixelError = std::accumulate(begin(landmarkPixelError), std::end(landmarkPixelError), 0.0f);
			string landmarkDetResult = lexical_cast<string>(landmarkNames.size()) + " landmarks detected. Average error in pixel: " + lexical_cast<string>(pixelError/landmarkNames.size());
			return landmarkDetResult;
		};
	private:
		string detectionTypeToString(DetectionType type) {
			switch (type) {
			case imageStatistic::DetectionType::TACC:
				return "TACC";
				break;
			case imageStatistic::DetectionType::FACC:
				return "FACC";
				break;
			case imageStatistic::DetectionType::TREJ:
				return "TREJ";
				break;
			case imageStatistic::DetectionType::FREJ:
				return "FREJ";
				break;
			case imageStatistic::DetectionType::NOGROUNDTRUTH:
				return "NOGROUNDTRUTH";
				break;
			default:
				return "Error";
				break;
			}
		};
	};
	map<path, imageStatistic> detectionResults;

	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	while(labeledImageSource->next()) {
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + labeledImageSource->getName().string());
		img = labeledImageSource->getImage();
	
		// Do the face-detection:
		vector<shared_ptr<ClassifiedPatch>> facePatches;
		for(const auto& detector : faceDetectors) {
			ImageLoggers->getLogger("detection").setCurrentImageName(labeledImageSource->getName().stem().string() + "_" + detector.first);
			facePatches = detector.second->detect(img);

			// For now, only work with 1 detector and the static facebox. Later:
			//    - allow left/right profile detectors
			//    - be smarter than just using the max-facebox.

		} // end for each face detector

		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		appLogger.debug("Finished face-detection. Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");
		shared_ptr<imageio::RectLandmark> detectedFace;
		if (facePatches.size() > 0) {
			detectedFace = make_shared<imageio::RectLandmark>("face", facePatches[0]->getPatch()->getBounds());
		}		

		// Create the binary mask (ROI) for the feature detectors:
		//Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
		//mask = patchToMask(facePatches[0]->getPatch(), mask);
		
		// Use only a single Rect ROI for now:
		// Using detect(...) with facePatches[0]->getPatch()->getBounds()

		// Detect all features in the face-box:
		map<string, shared_ptr<imageprocessing::Patch>> resultLms;
		if (facePatches.size() > 1 && featureDetectors.size() > 0) {
		
			Mat ffdMaxPosImg = img.clone();
			map<string, vector<shared_ptr<ClassifiedPatch>>> allFeaturePatches;
			for (const auto& detector : featureDetectors) {
				ImageLoggers->getLogger("detection").setCurrentImageName(labeledImageSource->getName().stem().string() + "_" + detector.first);
				Rect faceRoi = facePatches[0]->getPatch()->getBounds();
				faceRoi.height += 20;
				vector<shared_ptr<ClassifiedPatch>> resultingPatches = detector.second->detect(img, faceRoi);

				allFeaturePatches.insert(make_pair(detector.second->landmark, resultingPatches)); // be careful whether we want to use detector.first (its name) or detector.second->landmark
				if (resultingPatches.size() > 0) {
					shared_ptr<ModelLandmark> maxPos = make_shared<ModelLandmark>(detector.second->landmark, resultingPatches[0]->getPatch()->getX(), resultingPatches[0]->getPatch()->getY());
					maxPos->draw(ffdMaxPosImg);
				}
			}
			// Log the image with the max positive of every feature
			ImageLogger appImageLogger = ImageLoggers->getLogger("app");
			appImageLogger.setCurrentImageName(labeledImageSource->getName().stem().string());
			appImageLogger.intermediate(ffdMaxPosImg, doNothing, "AllFfpMaxPos");

			if (enableFeaturePointValidation) {
				// Tmp: Convert it to current map<string, vector<shared_ptr<imageprocessing::Patch>>> format
				map<string, vector<shared_ptr<imageprocessing::Patch>>> landmarkData2;
				for (const auto& feature : allFeaturePatches) {
					vector<shared_ptr<imageprocessing::Patch>> tmp;
					for (const auto& patch : feature.second) {
						tmp.push_back(patch->getPatch());
					}
					landmarkData2.insert(make_pair(feature.first, tmp));
					appLogger.debug(feature.first + " #cand: " + lexical_cast<string>(feature.second.size()));
				}

				featurePointsModel.setLandmarks(landmarkData2); // Should better use .run(landmarkData2); Clarity etc
				Mat rnsacImg = img.clone();
				resultLms = featurePointsModel.run(rnsacImg, 15.0f, numRansacIterations); // It would somehow be helpful to have a LandmarkSet data-type, consisting of #n strings and each with #m Patches, and having delete, add, ... operations. Can we do this with only the STL? (probably)
				if (resultLms.empty()) {
					// a few possibilities:
					// - If this happens, we just use the box from the FD
					// - we could count it as a failure of ransac (assuming the face-box was found correctly)
					// - future work: treat it as a legitimate failure and use another face-box!
					// * todo: the ransac algorithm should return only sets that are 4 LMs or bigger. Check if it adheres to that.
				}
				// set the improved face-box:
				if (!resultLms.empty()) {
					Mat ransacImage = img.clone();
					Rect detectedFaceRansacRect = featurePointsModel.evaluatorGetFaceCenter(resultLms, ransacImage);
					detectedFace = make_shared<imageio::RectLandmark>("face", detectedFaceRansacRect);
				}

			} else { // don't use ransac, just use the max-positive landmarks and the face-box from the FD
				for (const auto& feature : allFeaturePatches) {
					if (feature.second.size() > 0) {
						resultLms.insert(make_pair(feature.first, feature.second[0]->getPatch()));
					}
					appLogger.debug(feature.first + " #cand: " + lexical_cast<string>(feature.second.size()));
				}
			}
			
			for (const auto& lm : resultLms) {
				drawFfpsCircle(img, make_pair(lm.first, Point2f(lm.second->getX(), lm.second->getY())));
				drawFfpsText(img, make_pair(lm.first, Point2f(lm.second->getX(), lm.second->getY())));
				//imwrite("C:/Users/Patrik/Documents/Github/RANSAC.png", img);
				imageio::ModelLandmark l(lm.first, lm.second->getX(), lm.second->getY());
				l.draw(img);
			}
			appImageLogger.setCurrentImageName(labeledImageSource->getName().stem().string());
			appImageLogger.intermediate(img, doNothing, "AllFfpRansacBest");

		}

		LandmarkCollection groundtruth = labeledImageSource->getLandmarks();
		for (const auto& lm : groundtruth.getLandmarks()) {
			lm->draw(img, Scalar(0.0, 255.0, 0.0));
			drawFfpsText(img, make_pair(lm->getName(), lm->getPoint2D()));
		}

		// ImageLogger has the Draw etc functions for Patches... and landmarks also have draw...
		// The detectors always log face-boxes. But they should log boxes when it's a face and landmark-points when it's a landmark. Think about how to solve this, together with the other problems.
		// Why does the landmark class work with Vec3f and not Point3f?
		// Landmark class only needed for a) logging b) comparing & evaluation. (?) (search Peter's code)
		
		end = std::chrono::system_clock::now();
		elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		appLogger.info("Finished processing " + labeledImageSource->getName().string() + ". Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");
		
		imageStatistic stats;
		stats.numFaceCandidates = facePatches.size();
		if (!detectedFace) { // the facePatches is empty, i.e. this pointer is null
			stats.faceDetectionResult = imageStatistic::DetectionType::FREJ;
		} else { // we detected a face
			// check if it's the right face:
			try {
				shared_ptr<imageio::Landmark> face = groundtruth.getLandmark("face");
				face->draw(img, Scalar(0.0, 255.0, 0.0));
				detectedFace->draw(img, Scalar(0.0, 0.0, 255.0));
				bool isClose = detectedFace->isClose(*face.get(), 0.3f);
				if (isClose) {
					stats.faceDetectionResult = imageStatistic::DetectionType::TACC;
				} else {
					stats.faceDetectionResult = imageStatistic::DetectionType::FACC;
				}
				stats.haveFaceGroundtruth = true;
			} catch (invalid_argument& e) { // no face ground truth available
				// do nothing for now. Later: In case we run FFD, we could estimate the face box and compare again!
				stats.faceDetectionResult = imageStatistic::DetectionType::NOGROUNDTRUTH;
			}

			// Try to get the inner-eye distance, we'll use it later: (this can fail, for bigger poses, even the ground truth eye landmarks might not be available)
			try {
				shared_ptr<imageio::Landmark> l = groundtruth.getLandmark("left.eye.pupil.center");
				shared_ptr<imageio::Landmark> r = groundtruth.getLandmark("right.eye.pupil.center");
				stats.interEyeDistance = cv::norm(l->getPosition2D(), r->getPosition2D(), cv::NORM_L2);
			} catch (std::invalid_argument& e) {
				appLogger.info("No ground truth for the eye centers available, not calculating the inter-eye distance.");
				stats.interEyeDistance = 0.0f;
			}
			// Loop through our detected landmarks (top candidate), calculate the error w.r.t. the ground truth
			// What if we detect a LM (false-positive), but it's not in the ground truth?
			for (const auto& lm : resultLms) {
				shared_ptr<imageprocessing::Patch> p = lm.second;
				shared_ptr<imageio::ModelLandmark> detected = make_shared<imageio::ModelLandmark>(lm.first, p->getX(), p->getY());
				try {
					shared_ptr<imageio::Landmark> gt = groundtruth.getLandmark(lm.first); // throws when LM not found
					stats.landmarkNames.push_back(lm.first);
					float pixelError = cv::norm(detected->getPosition2D(), gt->getPosition2D(), cv::NORM_L2);
					stats.landmarkPixelError.emplace_back(pixelError);
					if (stats.interEyeDistance > 0.0f) { // we could/should use detected->isClose(gt, ied?); ? Or maybe calculate that later?
						if (pixelError <= stats.interEyeDistance/(10.0f)) { // is the error less than 10% of the ied? (should be <5 or something)
							stats.landmarkDetectionResult.push_back(imageStatistic::DetectionType::TACC);
						} else {
							stats.landmarkDetectionResult.push_back(imageStatistic::DetectionType::FACC);
						}
					} else { // no IED available, just use a static value of 10 (?) pixel
						// TODO
						stats.landmarkDetectionResult.push_back(imageStatistic::DetectionType::TACC);
					}
				} catch(invalid_argument& e) { // a ground truth landmark was not found
					// just don't record the stats for it
				}
			}
		}
		vector<RectLandmark> faces;
		vector<float> scores;	
		if (facePatches.size() > 0) {
			// Todo: We should use "detectedFace" instead of facePatches to make use of the ransac output.
			// However, with the new ransac face-box, we lose the score. (we could either A) juse use the old FD box score B) Run the FD again on the ransac-box C) calculate a new score based on some ransac criterion)
			// all posPatches (new NMS):
			for (const auto& p : facePatches) {
				faces.push_back(imageio::RectLandmark("face", p->getPatch()->getBounds()));
				scores.push_back(p->getProbability());
			}
			// only the maxPos patch: (normal OE)
			//faces.push_back(imageio::RectLandmark("face", facePatches[0]->getPatch()->getBounds()));
			//scores.push_back(facePatches[0]->getProbability());
		}
		landmarkSink.add(labeledImageSource->getName().string(), faces, scores);

		detectionResults.insert(make_pair(labeledImageSource->getName(), stats));
		// log stats for this image
		appLogger.info(stats.getFaceStatisticsString());
		appLogger.info(stats.getLandmarksStatisticsString());

	}
	landmarkSink.close();
	//Log stats for all images
	size_t numImages = labeledImageSource->getNames().size();
	int totalFacc = 0;
	int totalFrej = 0;
	int totalTacc = 0;
	int totalTrej = 0;
	int totalNoGroundtruth = 0;
	for (const auto& r : detectionResults) {
		switch (r.second.faceDetectionResult) {
		case imageStatistic::DetectionType::TACC:
			++totalTacc;
			break;
		case imageStatistic::DetectionType::FACC:
			++totalFacc;
			break;
		case imageStatistic::DetectionType::TREJ:
			++totalTrej;
			break;
		case imageStatistic::DetectionType::FREJ:
			++totalFrej;
			break;
		case imageStatistic::DetectionType::NOGROUNDTRUTH:
			++totalNoGroundtruth;
			break;
		default:
			appLogger.warn("DetectionType not set for this image. Something probably went wrong.");
			break;
		}
	}
	appLogger.info("Face detection result: Total images: " + lexical_cast<string>(numImages) + ". TACC: " + lexical_cast<string>(totalTacc) + ", FACC: " + lexical_cast<string>(totalFacc) + ", FREJ: " + lexical_cast<string>(totalFrej) + ", TREJ: " + lexical_cast<string>(totalTrej) + ".");

	return 0;
}

	// TODO important:
	// getPatchesROI Bug bei skalen, schraeg verschoben (?) bei x,y=0, s=1 sichtbar. No, I think I looked at this with MR, and the code was actually correct?
// Copy and = c'tors
	// pub/private
	// ALL in RegressorWVR.h/cpp is the same as in DetWVM! Except the classify loop AND threshold loading. -> own class (?)
	// Logger.drawscales
	// Logger draw 1 scale only, and points with color instead of boxes
	// logger filter lvls etc
	// problem when 2 diff. featuredet run on same scale
	// results dir from config etc
	// Diff. patch sizes: Cascade is a VDetectorVM, and calculates ONE subsampfac for the master-detector in his size. Then, for second det with diff. patchsize, calc remaining pyramids.
	// Test limit_reliability (SVM)	
	// Draw FFPs in different colors, and as points (symbols), not as boxes. See lib MR
	// Bisschen durcheinander mit pyramid_widths, subsampfac. Pyr_widths not necessary anymore? Pyr_widths are per detector
//  WVM/R: bisschen viele *thresh*...?
// wie verhaelt sich alles bei GRAY input image?? (imread, Logger)

// Error handling when something (eg det, img) not found -> STOP

// FFP-App: Read master-config. (Clean this up... keine vererbung mehr etc). FD. Then start as many FFD Det's as there are in the configs.

// @MR: Warum "-b" ? ComparisonRegr.xlsx 6grad systemat. fehler da ML +3.3, MR -3.3


/* 
/	Todo:
	* .lst: #=comment, ignore line
	* DetID alles int machen. Und dann mapper von int zu String (wo sich jeder Det am anfang eintraegt)
	* CascadeWvmOeSvmOe is a VDetVec... and returnFilterSize should return wvm->filtersizex... etc
	* I think the whole det-naming system ["..."] collapses when someone uses custom names (which we have to when using features)
/	* Filelists
	* optimizations (eg const)
/	* dump_BBList der ffp
	* OE: write field in patch, fout=1 -> passed, fout=0 failed OE
	* RVR/RVM
	* Why do we do (SVM)
	this->support[is][y*filter_size_x+x] = (unsigned char)(255.0*matdata[k++]);	 // because the training images grey level values were divided by 255;
	  but with the WVM, support is all float instead of uchar.

	 * erasing from the beginning of a vector is a slow operation, because at each step, all the elements of the vector have to be shifted down one place. Better would be to loop over the vector freeing everything (then clear() the vector. (or use a list, ...?) Improve speed of OE
	 * i++ --> ++i (faster)
*/


		/*cv::Mat color_img, color_hsv;
		int h_ = 0;   // H : 0 179, Hue
int s_ = 255; // S : 0 255, Saturation
int v_ = 255; // V : 0 255, Brightness Value
	const char *window_name = "HSV color";
	cv::namedWindow(window_name);
	cv::createTrackbar("H", window_name, &h_, 180, NULL, NULL);
	cv::createTrackbar("S", window_name, &s_, 255, NULL, NULL);
	cv::createTrackbar("V", window_name, &v_, 255, NULL, NULL);

	while(true) {
		color_hsv = cv::Mat(cv::Size(320, 240), CV_8UC3, cv::Scalar(h_,s_,v_));
		cv::cvtColor(color_hsv, color_img, CV_HSV2BGR);
		cv::imshow(window_name, color_img);
		int c = cv::waitKey(10);
		if (c == 27) break;
	}
	cv::destroyAllWindows();*/
