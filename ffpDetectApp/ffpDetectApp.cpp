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
#include "classification/ProbabilisticSvmClassifier.hpp"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/HistEq64Filter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"

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
	bool useFileList = false;
	bool useImgs = false;
	bool useDirectory = false;
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
			("config,c", po::value<path>()->required(), 
				"path to a config (.cfg) file")
			("input-list,l", po::value<path>(), 
				"input from a file containing a list of images")
			("input-file,f", po::value<vector<path>>(),
				"input one or several images")
			("input-dir,d", po::value<path>(),
				"input all images inside the directory")
			("output-dir,o", po::value<path>(),
				"output directory for the result images")
        ;

        po::positional_options_description p;
        p.add("input-file", -1);
        
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
		if (vm.count("input-list"))
		{
			useFileList = true;
			inputFilelist = vm["input-list"].as<path>();
		}
		if (vm.count("input-file"))
		{
			useImgs = true;
			inputFilenames = vm["input-file"].as<vector<path>>();
		}
		if (vm.count("input-dir"))
		{
			useDirectory = true;
			inputDirectory = vm["input-dir"].as<path>();
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
	
	Loggers->getLogger("classification").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("imageprocessing").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("detection").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("ffpDetectApp").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("ffpDetectApp");

	appLogger.debug("Verbose level for console output: " + logging::loglevelToString(logLevel));
	appLogger.debug("Using config: " + configFilename.string());
	appLogger.debug("Using output directory: " + outputPicsDir.string());
	if(outputPicsDir.empty()) {
		appLogger.info("Output directory not set. Writing images into current directory.");
	}

	ImageLoggers->getLogger("detection").addAppender(make_shared<imagelogging::ImageFileWriter>(imagelogging::loglevel::INFO, outputPicsDir));

	int numInputs = 0;
	if(useFileList==true) {
		numInputs++;
		appLogger.info("Using file-list as input: " + inputFilelist.string());
		shared_ptr<ImageSource> fileListImgSrc;
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
		numInputs++;
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
		numInputs++;
		appLogger.info("Using input images from directory: " + inputDirectory.string());
		try {
			imageSource = make_shared<DirectoryImageSource>(inputDirectory.string());
		} catch(const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
	}
	if(numInputs!=1) {
		appLogger.error("Please either specify a file-list (-l), an input-file (-f) or a directory (-d) (and only one of them) to run the program!");
		return EXIT_FAILURE;
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

	unordered_map<string, shared_ptr<FiveStageSlidingWindowDetector>> faceDetectors;
	unordered_map<string, shared_ptr<FiveStageSlidingWindowDetector>> featureDetectors;

	try {
		ptree ptDetectors = pt.get_child("detectors");
		for_each(begin(ptDetectors), end(ptDetectors), [&faceDetectors, &featureDetectors](ptree::value_type kv) {
			// TODO: 1) error handling if a key doesn't exist
			//		 2) make more dynamic, search for wvm or svm and add respective to map, or even just check if it's a WVM
			//				This would require another for_each and check if kv.first is 'wvm' or a .get...?
			ptree wvm = kv.second.get_child("wvm");	// TODO add error checking
			ptree svm = kv.second.get_child("svm");	// TODO add error checking
			ptree imgpyr = kv.second.get_child("pyramid");	// TODO add error checking
			ptree oeCfg = kv.second.get_child("overlapElimination");	// TODO add error checking. get_child throws when child doesn't exist.
			string landmarkName = kv.second.get<string>("landmark");

			shared_ptr<ProbabilisticWvmClassifier> pwvm = ProbabilisticWvmClassifier::loadConfig(wvm);
			shared_ptr<ProbabilisticSvmClassifier> psvm = ProbabilisticSvmClassifier::loadConfig(svm);

			//pwvm->getWvm()->setLimitReliabilityFilter(-0.5f);
			//psvm->getSvm()->setThreshold(-1.0f);	// TODO read this from the config

			shared_ptr<OverlapElimination> oe = make_shared<OverlapElimination>(oeCfg.get<float>("dist", 5.0f), oeCfg.get<float>("ratio", 0.0f));
			
			// This:
			shared_ptr<ImagePyramid> imgPyr = make_shared<ImagePyramid>(imgpyr.get<float>("minScaleFactor", 0.09f), imgpyr.get<float>("maxScaleFactor", 0.25f), imgpyr.get<float>("incrementalScaleFactor", 0.9f));
			imgPyr->addImageFilter(make_shared<GrayscaleFilter>());
			shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(imgPyr, imgpyr.get<int>("patch.width"), imgpyr.get<int>("patch.height"));
			// Or:
			//shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(config.get<int>("pyramid.patch.width"), config.get<int>("pyramid.patch.height"), config.get<int>("pyramid.patch.minWidth"), config.get<int>("pyramid.patch.maxWidth"), config.get<double>("pyramid.scaleFactor"));
			//featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());

			featureExtractor->addPatchFilter(make_shared<HistEq64Filter>());

			shared_ptr<SlidingWindowDetector> det = make_shared<SlidingWindowDetector>(pwvm, featureExtractor);

			shared_ptr<FiveStageSlidingWindowDetector> fsd = make_shared<FiveStageSlidingWindowDetector>(det, oe, psvm);
			fsd->landmark = landmarkName;
			if (landmarkName == "face")	{
				faceDetectors.insert(make_pair(kv.first, fsd));
			} else {
				featureDetectors.insert(make_pair(kv.first, fsd));
			}
		});
	} catch(const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	} catch (const invalid_argument& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	} catch (const runtime_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	// lm-loading
	// output-dir
	// load ffd/ROI
	// relative bilder-pfad aus filelist
	// our libs: add library dependencies (eg to boost) in add_library ?
	// -f file.png fails?

	/* Note: We could change/write/add something to the config with
	pt.put("detection.svm.threshold", -0.5f);
	If the value already exists, it gets overwritten, if not, it gets created.
	Save it with:
	write_info("C:\\Users\\Patrik\\Documents\\GitHub\\ffpDetectApp.cfg", pt);
	*/

	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	while(imageSource->next()) {

		img = imageSource->getImage();
		start = std::chrono::system_clock::now();
		
		for(auto detector : faceDetectors) {

			// The original image
			ImageLoggers->getLogger("detection").setCurrentImageName(imageSource->getName().stem().string() + "_" + detector.first);
			vector<shared_ptr<ClassifiedPatch>> resultingPatches = detector.second->detect(img);

			/*
			// WVM stage
			vector<shared_ptr<ClassifiedPatch>> resultingPatches = det->detect(img);
			Mat imgWvm = img.clone();
			drawFaceBoxes(imgWvm, resultingPatches);
			cv::imwrite(path("").string() + "_wvm.png", imgWvm);

			function <void ()> f = bind(drawFaceBoxes, img, resultingPatches);
			logtest("test", img, f);

			// WVM OE stage
			resultingPatches = oe->eliminate(resultingPatches);
			Mat imgWvmOe = img.clone();
			drawFaceBoxes(imgWvmOe, resultingPatches);
			cv::imwrite(path("").string() + "_wvmoe.png", imgWvmOe);

			// SVM stage
			vector<shared_ptr<ClassifiedPatch>> svmPatches;
			for(const auto& patch : resultingPatches) {
				svmPatches.push_back(make_shared<ClassifiedPatch>(patch->getPatch(), psvm->classify(patch->getPatch()->getData())));
			}
			Mat imgSvmAll = img.clone();
			drawFaceBoxes(imgSvmAll, svmPatches);
			cv::imwrite(path("").string() + "_svmall.png", imgSvmAll);

			// Only the positive SVM patches
			vector<shared_ptr<ClassifiedPatch>> svmPatchesPositive;
			for(const auto& patch : svmPatches) {
				if(patch->isPositive()) {
					svmPatchesPositive.push_back(patch);
				}
			}
			Mat imgSvmPos = img.clone();
			drawFaceBoxes(imgSvmPos, svmPatchesPositive);
			cv::imwrite(path("").string() + "_svmpos.png", imgSvmPos);

			// The highest one of all the positively classified SVM patches
			Mat imgSvmMaxPos = img.clone();
			if(svmPatchesPositive.size()>0) {
				sort(make_indirect_iterator(svmPatchesPositive.begin()), make_indirect_iterator(svmPatchesPositive.end()), greater<ClassifiedPatch>()); // Careful, this invalidates all copies of svmPatchesPositive!
				vector<shared_ptr<ClassifiedPatch>> svmPatchesMaxPositive;
				svmPatchesMaxPositive.push_back(svmPatchesPositive[0]);
				drawFaceBoxes(imgSvmMaxPos, svmPatchesMaxPositive);
			}
			cv::imwrite(path("").string() + "_svmmaxpos.png", imgSvmMaxPos);
			*/
		} // end for each face detector

		end = std::chrono::system_clock::now();

		int elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);

		stringstream ss;
		ss << std::ctime(&end_time);
		appLogger.info("finished computation at " + ss.str() + "elapsed time: " + lexical_cast<string>(elapsed_seconds) + "s, " + lexical_cast<string>(elapsed_mseconds) + "ms\n");

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
