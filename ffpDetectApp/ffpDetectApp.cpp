/*
 * ffpDetectApp.cpp
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

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/iterator/indirect_iterator.hpp"
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
#include "imageio/LandmarkCollection.hpp"
#include "imageio/RectLandmark.hpp"
#include "imageio/DefaultLandmarkSink.hpp"

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
using boost::make_indirect_iterator;
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
	vector<path> inputPaths;
	path inputFilelist;
	path inputDirectory;
	vector<path> inputFilenames;
	path configFilename;
	shared_ptr<ImageSource> imageSource;
	path outputPicsDir;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("verbose-images,w", po::value<string>(&verboseLevelImages)->implicit_value("INTERMEDIATE")->default_value("FINAL","write images with FINAL loglevel or below."),
				  "specify the verbosity of the image output: FINAL, INTERMEDIATE, INFO, DEBUG or TRACE")
			("config,c", po::value<path>()->required(), 
				"path to a config (.cfg) file")
			("input,i", po::value<vector<path>>()->required(), 
				"input from one or more files, a directory, or a  .lst-file containing a list of images")
			("output-dir,o", po::value<path>()->default_value("."),
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
		if (vm.count("verbose")) {
			verboseLevelConsole = vm["verbose"].as<string>();
		}
		if (vm.count("verbose-images")) {
			verboseLevelImages = vm["verbose-images"].as<string>();
		}
		if (vm.count("input"))
		{
			inputPaths = vm["input"].as<vector<path>>();
		}
		if (vm.count("config"))
		{
			configFilename = vm["config"].as<path>();
		}
		if (vm.count("output-dir"))
		{
			outputPicsDir = vm["output-dir"].as<path>();
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

	if(inputPaths.size() > 1) {
		// We assume the user has given several, valid images
		useImgs = true;
		inputFilenames = inputPaths;
	} else if (inputPaths.size() == 1) {
		// We assume the user has given either an image, directory, or a .lst-file
		if (inputPaths[0].extension().string() == ".lst") { // check for .lst first
			useFileList = true;
			inputFilelist = inputPaths[0];
		} else if (boost::filesystem::is_directory(inputPaths[0])) { // check if it's a directory
			useDirectory = true;
			inputDirectory = inputPaths[0];
		} else { // it must be an image
			useImgs = true;
			inputFilenames = inputPaths;
		}
	} else {
		appLogger.error("Please either specify one or several files, a directory, or a .lst-file containing a list of images to run the program!");
		return EXIT_FAILURE;
	}


	if(useFileList==true) {
		appLogger.info("Using file-list as input: " + inputFilelist.string());
		shared_ptr<ImageSource> fileListImgSrc; // TODO VS2013 change to unique_ptr, rest below also
		try {
			fileListImgSrc = make_shared<FileListImageSource>(inputFilelist.string());
		} catch(const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
		//shared_ptr<DidLandmarkFormatParser> didParser= make_shared<DidLandmarkFormatParser>();
		//vector<path> landmarkDir; landmarkDir.push_back(path("C:\\Users\\Patrik\\Github\\data\\labels\\xm2vts\\guosheng\\"));
		//shared_ptr<DefaultNamedLandmarkSource> lmSrc = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(fileImgSrc, ".did", GatherMethod::ONE_FILE_PER_IMAGE_DIFFERENT_DIRS, landmarkDir), didParser);
		//imageSource = make_shared<NamedLabeledImageSource>(fileImgSrc, lmSrc);
		imageSource = fileListImgSrc;
	}
	if(useImgs==true) {
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
		//shared_ptr<DidLandmarkFormatParser> didParser= make_shared<DidLandmarkFormatParser>();
		//vector<path> landmarkDir; landmarkDir.push_back(path("C:\\Users\\Patrik\\Github\\data\\labels\\xm2vts\\guosheng\\"));
		//shared_ptr<DefaultNamedLandmarkSource> lmSrc = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(fileImgSrc, ".did", GatherMethod::ONE_FILE_PER_IMAGE_DIFFERENT_DIRS, landmarkDir), didParser);
		//imageSource = make_shared<NamedLabeledImageSource>(fileImgSrc, lmSrc);
		imageSource = fileImgSrc;
	}
	if(useDirectory==true) {
		appLogger.info("Using input images from directory: " + inputDirectory.string());
		try {
			imageSource = make_shared<DirectoryImageSource>(inputDirectory.string());
		} catch(const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
	}

	const float DETECT_MAX_DIST_X = 0.33f;	// --> Config / Landmarks
	const float DETECT_MAX_DIST_Y = 0.33f;
	const float DETECT_MAX_DIFF_W = 0.33f;

	int TOT = 0;
	int TACC = 0;
	int FACC = 0;
	int NOCAND = 0;
	int DONTKNOW = 0;

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

				//pwvm->getWvm()->setLimitReliabilityFilter(-0.5f);
				//psvm->getSvm()->setThreshold(-1.0f);	// TODO read this from the config

				shared_ptr<OverlapElimination> oe = make_shared<OverlapElimination>(oeCfg.get<float>("dist", 5.0f), oeCfg.get<float>("ratio", 0.0f));

				// This:
				shared_ptr<ImagePyramid> imgPyr = make_shared<ImagePyramid>(imgpyr.get<float>("incrementalScaleFactor", 0.9f), imgpyr.get<float>("minScaleFactor", 0.09f), imgpyr.get<float>("maxScaleFactor", 0.25f));
				imgPyr->addImageFilter(make_shared<GrayscaleFilter>());
				shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(imgPyr, imgpyr.get<int>("patch.width"), imgpyr.get<int>("patch.height"));
				// Or:
				//shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(config.get<int>("pyramid.patch.width"), config.get<int>("pyramid.patch.height"), config.get<int>("pyramid.patch.minWidth"), config.get<int>("pyramid.patch.maxWidth"), config.get<double>("pyramid.scaleFactor"));
				//featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());

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
				
				//psvm->getSvm()->setThreshold(-1.0f);	// TODO read this from the config

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

	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	while(imageSource->next()) {
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + imageSource->getName().string());
		img = imageSource->getImage();
		
		// Do the face-detection:
		vector<shared_ptr<ClassifiedPatch>> facePatches;
		for(const auto& detector : faceDetectors) {
			ImageLoggers->getLogger("detection").setCurrentImageName(imageSource->getName().stem().string() + "_" + detector.first);
			facePatches = detector.second->detect(img);

			// For now, only work with 1 detector and the static facebox. Later:
			//    - allow left/right profile detectors
			//    - be smarter than just using the max-facebox.

			if (facePatches.size()>0) {
				imageio::LandmarkCollection lmc;
				shared_ptr<imageio::RectLandmark> mostProbablyFace = make_shared<imageio::RectLandmark>(detector.first, facePatches[0]->getPatch()->getBounds());
				lmc.insert(mostProbablyFace);

				imageio::DefaultLandmarkSink lms("Y:\\raidbuf\\lfpw_landmarks_patrik\\");
				lms.write(lmc, imageSource->getName().stem().string());

			}
			

		} // end for each face detector

		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		appLogger.debug("Finished face-detection. Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");
		
		// Create the binary mask (ROI) for the feature detectors:
		//Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
		//mask = patchToMask(facePatches[0]->getPatch(), mask);
		
		// Use only a single Rect ROI for now:
		// Using detect(...) with facePatches[0]->getPatch()->getBounds()

		// Detect all features in the face-box:
		map<string, vector<shared_ptr<ClassifiedPatch>>> allFeaturePatches;
		for (const auto& detector : featureDetectors) {
			ImageLoggers->getLogger("detection").setCurrentImageName(imageSource->getName().stem().string() + "_" + detector.first);
			vector<shared_ptr<ClassifiedPatch>> resultingPatches = detector.second->detect(img, facePatches[0]->getPatch()->getBounds());

			//shared_ptr<PyramidFeatureExtractor> pfe = detector.second->getPyramidFeatureExtractor();
			//drawScales(ffdResultImg, 20, 20, pfe->getMinScaleFactor(), pfe->getMaxScaleFactor());
			allFeaturePatches.insert(make_pair(detector.second->landmark, resultingPatches)); // be careful if we want to use detector.first (its name) or detector.second->landmark
		}


		/*
		//Mat ffdResultImg = img.clone();
		for (const auto& features : allFeaturePatches) {
			for (const auto& patch : features.second) {
				// patch to landmark
				// add a function in libDetection, either Patch.getLandmark or Helper...::PatchToLandmark(...) (not ClassifiedPatch!)
				// Anmerkung von peter: libDetection/libImageProcessing hat bisher keine Abhaengigkeit von libImageIO, aber Patch.getLandmark/ClassifiedPatch.getLandmark wuerde dazu fuehren
				// Anmerkung von peter: (ebenso umgekehrt, wenn der Helper in libImageIO laege - die Abhaengigkeit (von libImageProcessing) waere dann sogar sehr unschoen)
				RectLandmark lm(features.first, patch->getPatch()->getX(), patch->getPatch()->getY(), patch->getPatch()->getWidth(), patch->getPatch()->getHeight());
				lm.draw(ffdResultImg);
			}
			Mat tmp = img.clone();
			drawFaceBoxes(tmp, features.second);

		}
		*/

		
		end = std::chrono::system_clock::now();
		elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		appLogger.info("Finished processing " + imageSource->getName().string() + ". Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");

		TOT++;
		vector<string> resultingPatches;
		if(resultingPatches.size()<1) {
			//std::cout << "[ffpDetectApp] No face-candidates at all found:  " << filenames[i] << std::endl;
			NOCAND++;
		} else {
			// TODO Check if the LM exists or it will crash! Currently broken!
			if(false) { //no groundtruth
				//std::cout << "[ffpDetectApp] No ground-truth available, not counting anything: " << filenames[i] << std::endl;
				++DONTKNOW;
			} else { //we have groundtruth
				/*int gt_w = groundtruthFaceBoxes[i].getWidth();
				int gt_h = groundtruthFaceBoxes[i].getHeight();
				int gt_cx = groundtruthFaceBoxes[i].getX();
				int gt_cy = groundtruthFaceBoxes[i].getY();
				// TODO implement a isClose, isDetected... or something like that function
				if (abs(gt_cx - svmPatches[0]->getPatch()->getX()) < DETECT_MAX_DIST_X*(float)gt_w &&
					abs(gt_cy - svmPatches[0]->getPatch()->getY()) < DETECT_MAX_DIST_Y*(float)gt_w &&
					abs(gt_w - svmPatches[0]->getPatch()->getWidth()) < DETECT_MAX_DIFF_W*(float)gt_w       ) {
				
					std::cout << "[ffpDetectApp] TACC (1/1): " << filenames[i] << std::endl;
					TACC++;
				} else {
					std::cout << "[ffpDetectApp] Face not found, wrong position:  " << filenames[i] << std::endl;
					FACC++;
				}*/
			} // end no groundtruth
		}

		std::cout << std::endl;
		std::cout << "[ffpDetectApp] -------------------------------------" << std::endl;
		std::cout << "[ffpDetectApp] TOT:  " << TOT << std::endl;
		std::cout << "[ffpDetectApp] TACC:  " << TACC << std::endl;
		std::cout << "[ffpDetectApp] FACC:  " << FACC << std::endl;
		std::cout << "[ffpDetectApp] NOCAND:  " << NOCAND << std::endl;
		std::cout << "[ffpDetectApp] DONTKNOW:  " << DONTKNOW << std::endl;
		std::cout << "[ffpDetectApp] -------------------------------------" << std::endl;

	}

	std::cout << std::endl;
	std::cout << "[ffpDetectApp] =====================================" << std::endl;
	std::cout << "[ffpDetectApp] =====================================" << std::endl;
	std::cout << "[ffpDetectApp] TOT:  " << TOT << std::endl;
	std::cout << "[ffpDetectApp] TACC:  " << TACC << std::endl;
	std::cout << "[ffpDetectApp] FACC:  " << FACC << std::endl;
	std::cout << "[ffpDetectApp] NOCAND:  " << NOCAND << std::endl;
	std::cout << "[ffpDetectApp] DONTKNOW:  " << DONTKNOW << std::endl;
	std::cout << "[ffpDetectApp] =====================================" << std::endl;
	
	return 0;
}

	// My cmdline-arguments: -f C:\Users\Patrik\Documents\GitHub\data\firstrun\theRealWorld_png2.lst

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
