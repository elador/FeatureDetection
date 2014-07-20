/*
 * sdmTraining.cpp
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

#include "superviseddescent/LandmarkBasedSupervisedDescentTraining.hpp"
#include "superviseddescent/SdmLandmarkModel.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
using namespace superviseddescent;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using std::map;
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

// computation...normalization ... alignment ...

enum class FilterByFaceDetection { // on what to train the mean?
	NONE,
	VIOLAJONES // orig paper, sect. 3.1. & Fig. 2b)
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
				"output filename")
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
	Loggers->getLogger("sdmTraining").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("sdmTraining");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

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
	int numCascadeSteps; // How many cascade steps to learn? (i.e. how many regressors)
	vector<string> descriptorTypes;
	vector<shared_ptr<DescriptorExtractor>> descriptorExtractors;
	LandmarkBasedSupervisedDescentTraining::Regularisation regularisation;

	// Read the stuff from the config:
	ptree pt;
	try {
		read_info(configFilename.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(string("Error reading the config file: ") + error.what());
		return EXIT_FAILURE;
	}
	try {
		// Get stuff from the parameters subtree
		ptree ptParameters = pt.get_child("parameters");
		numSamplesPerImage = ptParameters.get<int>("numSamplesPerImage", 10);
		numCascadeSteps = ptParameters.get<int>("numCascadeSteps", 5);
		regularisation.factor = ptParameters.get<float>("regularisationFactor", 0.5f);
		regularisation.regulariseAffineComponent = ptParameters.get<bool>("regulariseAffineComponent", false);
		regularisation.regulariseWithEigenvalueThreshold = ptParameters.get<bool>("regulariseWithEigenvalueThreshold", false);
		// Read the 'featureDescriptors' sub-tree:
		ptree ptFeatureDescriptors = ptParameters.get_child("featureDescriptors");
		for (const auto& kv : ptFeatureDescriptors) {
			string descriptorType = kv.second.get<string>("descriptorType");
			string descriptorPostprocessing = kv.second.get<string>("descriptorPostprocessing", "none");
			string descriptorParameters = kv.second.get<string>("descriptorParameters", "");
			if (descriptorType == "OpenCVSift") { // Todo: make a load method in each descriptor
				shared_ptr<DescriptorExtractor> sift = std::make_shared<SiftDescriptorExtractor>();
				descriptorExtractors.push_back(sift);
			}
			else if (descriptorType == "vlhog-dt") {
				vector<string> params;
				boost::split(params, descriptorParameters, boost::is_any_of(" "));
				if (params.size() != 6) {
					throw std::logic_error("descriptorParameters must contain numCells, cellSize and numBins.");
				}
				int numCells = boost::lexical_cast<int>(params[1]);
				int cellSize = boost::lexical_cast<int>(params[3]);
				int numBins = boost::lexical_cast<int>(params[5]);
				shared_ptr<DescriptorExtractor> vlhogDt = std::make_shared<VlHogDescriptorExtractor>(VlHogDescriptorExtractor::VlHogType::DalalTriggs, numCells, cellSize, numBins);
				descriptorExtractors.push_back(vlhogDt);
			}
			else if (descriptorType == "vlhog-uoctti") {
				vector<string> params;
				boost::split(params, descriptorParameters, boost::is_any_of(" "));
				if (params.size() != 6) {
					throw std::logic_error("descriptorParameters must contain numCells, cellSize and numBins.");
				}
				int numCells = boost::lexical_cast<int>(params[1]);
				int cellSize = boost::lexical_cast<int>(params[3]);
				int numBins = boost::lexical_cast<int>(params[5]);
				shared_ptr<DescriptorExtractor> vlhogUoctti = std::make_shared<VlHogDescriptorExtractor>(VlHogDescriptorExtractor::VlHogType::Uoctti, numCells, cellSize, numBins);
				descriptorExtractors.push_back(vlhogUoctti);
			} else {
				throw std::logic_error("descriptorType does not match 'OpenCVSift', 'vlhog-dt' or 'vlhog-uoctti'.");
			}
			descriptorTypes.push_back(descriptorType);
		}

		// Get stuff from the modelLandmarks subtree:
		ptree ptModelLandmarks = pt.get_child("modelLandmarks");
		modelLandmarkType = ptModelLandmarks.get<string>("landmarkType");
		appLogger.debug("Type of the model landmarks: " + modelLandmarkType);
		string modelLandmarksUsage = ptModelLandmarks.get<string>("landmarks");
		if (modelLandmarksUsage.empty()) {
			// value is empty, meaning it's a node and the user should specify a list of 'landmarks'
			ptree ptModelLandmarksList = ptModelLandmarks.get_child("landmarks");
			for (const auto& kv : ptModelLandmarksList) {
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
			appLogger.error("The path given is neither a directory nor a .lst/.txt-file containing a list of images: " + d.images.string());
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

	string faceDetectionModel("C:\\opencv\\opencv_2.4.8_prebuilt\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml");
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(faceDetectionModel))
	{
		cout << "Error loading face detection model." << endl;
		return EXIT_FAILURE;
	}

	// App-config:
	FilterByFaceDetection filterByFaceDetection = FilterByFaceDetection::VIOLAJONES;
	// Add a switch "use_GT_as_detectionResults" ?
	
	// Our data:
	vector<Mat> trainingImages;
	vector<Mat> trainingGroundtruthLandmarks;
	vector<Rect> trainingFaceboxes;

	start = std::chrono::system_clock::now();
	appLogger.info("Reading data, doing V&J if enabled...");

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
				trainingImages.push_back(img);
				trainingGroundtruthLandmarks.push_back(landmarks);
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
				trainingImages.push_back(img);
				trainingGroundtruthLandmarks.push_back(landmarks);
				trainingFaceboxes.push_back(detectedFaces[0]);
			}
		}
	}

	end = std::chrono::system_clock::now();
	int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	appLogger.debug("Finished after " + lexical_cast<string>(elapsed_mseconds)+"ms.");

	LandmarkBasedSupervisedDescentTraining tr;
	tr.setNumSamplesPerImage(numSamplesPerImage);
	tr.setNumCascadeSteps(numCascadeSteps);
	tr.setRegularisation(regularisation);
	tr.setAlignGroundtruth(LandmarkBasedSupervisedDescentTraining::AlignGroundtruth::NONE); // TODO Read from config!
	tr.setMeanNormalization(LandmarkBasedSupervisedDescentTraining::MeanNormalization::UNIT_SUM_SQUARED_NORMS); // TODO Read from config!
	SdmLandmarkModel model = tr.train(trainingImages, trainingGroundtruthLandmarks, trainingFaceboxes, modelLandmarks, descriptorTypes, descriptorExtractors);
	
	std::time_t currentTime_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	string currentTime = std::ctime(&currentTime_t);
	
	model.save(outputFilename, "Trained on " + currentTime.substr(0, currentTime.length() - 1));

	appLogger.info("Finished training. Saved model to " + outputFilename.string() + ".");

	return 0;
}
