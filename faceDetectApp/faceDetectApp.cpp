/*
 * faceDetectApp.cpp
 *
 *  Created on: 08.04.2013
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

#include "classification/RbfKernel.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/WvmClassifier.hpp"
#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/HistEq64Filter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"

#include "detection/SlidingWindowDetector.hpp"
#include "detection/ClassifiedPatch.hpp"
#include "detection/OverlapElimination.hpp"

#include "imageio/LandmarksHelper.hpp"
#include "imageio/LandmarkSource.hpp"
#include "imageio/LabeledImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/RepeatingFileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/DidLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
using namespace std;
using namespace imageprocessing;
using namespace detection;
using namespace classification;
using namespace imageio;
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;
using boost::make_indirect_iterator;
using boost::property_tree::ptree;
using boost::property_tree::info_parser::read_info;

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
    return os;
}

/**
 * Takes a list of classified patches and creates a single probability map of face region locations.
 *
 * Note/TODO: We could increase the probability of a region when nearby patches (in x/y and scale)
 *            also say it's a face. But that might be very difficult with this current approach.
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

void drawFaceBoxes(Mat image, vector<shared_ptr<ClassifiedPatch>> patches)
{
	for(auto pit = patches.begin(); pit != patches.end(); pit++) {
		shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
		shared_ptr<Patch> patch = classifiedPatch->getPatch();
		cv::rectangle(image, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
	}
}

void drawSccales()
{

}


void drawWindow(Mat image, string windowName, int windowX, int windowY)
{
	cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName, image);
	cvMoveWindow(windowName.c_str(), windowX, windowY);
	cv::waitKey(30);
}


int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(3759128);
	#endif
	
	int verbose_level_text;
	int verbose_level_images;
	bool useFileList = false;
	bool useImgs = false;
	bool useDirectory = false;
	string inputFilelist;
	string inputDirectory;
	vector<std::string> inputFilenames;
	shared_ptr<ImageSource> imageSource;
	
	try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("verbose-text,v", po::value<int>(&verbose_level_text)->implicit_value(2)->default_value(1,"minimal text output"),
                  "enable text-verbosity (optionally specify level)")
            ("verbose-images,w", po::value<int>(&verbose_level_images)->implicit_value(2)->default_value(1,"minimal image output"),
                  "enable image-verbosity (optionally specify level)")
            ("input-list,l", po::value<string>(), "input from a file containing a list of images")
            ("input-file,f", po::value<vector<string>>(), "input one or several images")
			("input-dir,d", po::value<string>(), "input all images inside the directory")
        ;

        po::positional_options_description p;
        p.add("input-file", -1);	// allow one or several -f directives
        
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).
                  options(desc).positional(p).run(), vm);
        po::notify(vm);
    
        if (vm.count("help")) {
            cout << "[faceDetectApp] Usage: options_description [options]\n";
            cout << desc;
            return 0;
        }
        if (vm.count("input-list"))
        {
            cout << "[faceDetectApp] Using file-list as input: " << vm["input-list"].as<string>() << "\n";
			useFileList = true;
			inputFilelist = vm["input-list"].as<string>();
        }
        if (vm.count("input-file"))
        {
            cout << "[faceDetectApp] Using input images: " << vm["input-file"].as<vector<string>>() << "\n";
			useImgs = true;
			inputFilenames = vm["input-file"].as< vector<string> >();
        }
		if (vm.count("input-dir"))
		{
			cout << "[faceDetectApp] Using input images: " << vm["input-dir"].as<string>() << "\n";
			useDirectory = true;
			inputDirectory = vm["input-dir"].as<string>();
		}
        if (vm.count("verbose-text")) {
            cout << "[faceDetectApp] Verbose level for text: " << vm["verbose-text"].as<int>() << "\n";
        }
        if (vm.count("verbose-images")) {
            cout << "[faceDetectApp] Verbose level for images: " << vm["verbose-images"].as<int>() << "\n";
        }
    }
    catch(std::exception& e) {
        cout << e.what() << "\n";
        return 1;
    }

	Loggers->getLogger("classification").addAppender(make_shared<logging::ConsoleAppender>(loglevel::TRACE));
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(loglevel::TRACE));

	int numInputs = 0;
	if(useFileList==true) {
		numInputs++;
		shared_ptr<ImageSource> fileImgSrc = make_shared<FileListImageSource>(inputFilelist);
		shared_ptr<DidLandmarkFormatParser> didParser= make_shared<DidLandmarkFormatParser>();
		vector<path> landmarkDir; landmarkDir.push_back(path("C:\\Users\\Patrik\\Github\\data\\labels\\xm2vts\\guosheng\\"));
		shared_ptr<LandmarkSource> lmSrc = make_shared<LandmarkSource>(LandmarkFileGatherer::gather(fileImgSrc, ".did", GatherMethod::ONE_FILE_PER_IMAGE_DIFFERENT_DIRS, landmarkDir), didParser);
		imageSource = make_shared<LabeledImageSource>(fileImgSrc, lmSrc);
	}
	if(useImgs==true) {
		numInputs++;
		//imageSource = make_shared<FileImageSource>(inputFilenames);
		//imageSource = make_shared<RepeatingFileImageSource>("C:\\Users\\Patrik\\GitHub\\data\\firstrun\\ws_8.png");
		shared_ptr<ImageSource> fileImgSrc = make_shared<FileImageSource>(inputFilenames);
		shared_ptr<DidLandmarkFormatParser> didParser= make_shared<DidLandmarkFormatParser>();
		vector<path> landmarkDir; landmarkDir.push_back(path("C:\\Users\\Patrik\\Github\\data\\labels\\xm2vts\\guosheng\\"));
		shared_ptr<LandmarkSource> lmSrc = make_shared<LandmarkSource>(LandmarkFileGatherer::gather(fileImgSrc, ".did", GatherMethod::ONE_FILE_PER_IMAGE_DIFFERENT_DIRS, landmarkDir), didParser);
		imageSource = make_shared<LabeledImageSource>(fileImgSrc, lmSrc);
	}
	if(useDirectory==true) {
		numInputs++;
		imageSource = make_shared<DirectoryImageSource>(inputDirectory);
	}
	if(numInputs!=1) {
		cout << "[faceDetectApp] Error: Please either specify a file-list, an input-file or a directory (and only one of them) to run the program!" << endl;
		return 1;
	}

	ptree pt;
	read_info("C:\\Users\\Patrik\\GitHub\\faceDetectApp.cfg", pt);		// TODO add check if file exists/throw

	ptree ptClassifiers = pt.get_child("classifiers");
	unordered_multimap<string, shared_ptr<ProbabilisticClassifier>> classifiers;
	for_each(begin(ptClassifiers), end(ptClassifiers), [&classifiers](ptree::value_type kv) {
		// TODO: 1) error handling if a key doesn't exist
		//		 2) make more dynamic, search for wvm or svm and add respective to map, or even just check if it's a WVM
		//				This would require another for_each and check if kv.first is 'wvm' or a .get...?
		ptree wvm = kv.second.get_child("wvm");	// TODO add error checking
		classifiers.insert(make_pair(kv.first + ".wvm", ProbabilisticWvmClassifier::loadConfig(wvm)));
	});

	shared_ptr<ImagePyramid> pyr = make_shared<ImagePyramid>(pt.get<float>("imagePyramid.minScaleFactor", 0.09f), pt.get<float>("imagePyramid.maxScaleFactor", 0.25f), pt.get<float>("imagePyramid.incrementalScaleFactor", 0.9f));	// (0.09, 0.25, 0.9) is nearly the same as old 90, 9, 0.9
	pyr->addImageFilter(make_shared<GrayscaleFilter>());
	shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(pyr, 20, 20);
	featureExtractor->addPatchFilter(make_shared<HistEq64Filter>());

	//shared_ptr<ProbabilisticWvmClassifier> test = dynamic_pointer_cast<ProbabilisticWvmClassifier>(classifiers.find("faceFrontal.wvm")->second);
	//test->getWvm()->setNumUsedFilters(280);

	unordered_multimap<string, shared_ptr<SlidingWindowDetector>> detectors;
	for(auto classifier : classifiers) {
		detectors.insert(make_pair(classifier.first, make_shared<SlidingWindowDetector>(classifier.second, featureExtractor)));
	}

	Mat img;

	while ((img = imageSource->get()).rows != 0) { // TODO Can we check against !=Mat() somehow? or .empty?

		cv::namedWindow("src", CV_WINDOW_AUTOSIZE); cv::imshow("src", img);
		cvMoveWindow("src", 0, 0);
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		int det=0;
		vector<Mat> regionProbMaps;
		for(auto detector : detectors) {
			vector<shared_ptr<ClassifiedPatch>> resultingPatches = detector.second->detect(img);
			Mat imgDet = img.clone();
			drawFaceBoxes(imgDet, resultingPatches);
			if(det==0) {
				drawWindow(imgDet, detector.first, 0, 0);
			} else if(det==1) {
				drawWindow(imgDet, detector.first, 500, 0);
			} else if(det==2) {
				drawWindow(imgDet, detector.first, 0, 500);
			} else if(det==3) {
				drawWindow(imgDet, detector.first, 500, 500);
			}

			Mat faceRegionProbabilityMap = getFaceRegionProbabilityMapFromPatchlist(resultingPatches, img.cols, img.rows);
			regionProbMaps.push_back(faceRegionProbabilityMap);

			++det;
		}
		
		Mat wholeFaceRegionProbabilityMap = Mat::zeros(regionProbMaps[0].rows, regionProbMaps[0].cols, CV_32FC1);
		for (const auto& map : regionProbMaps) {
			wholeFaceRegionProbabilityMap += map;
		}
		wholeFaceRegionProbabilityMap /= (float)regionProbMaps.size();
		wholeFaceRegionProbabilityMap *= 255.0f;
		wholeFaceRegionProbabilityMap.convertTo(wholeFaceRegionProbabilityMap, CV_8UC1);
		cv::namedWindow("pr", CV_WINDOW_AUTOSIZE); cv::imshow("pr", wholeFaceRegionProbabilityMap);
		cvMoveWindow("pr", 1000, 500);
		cv::waitKey(0);
		
		
		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);
		std::cout << "Elapsed time: " << elapsed_mseconds << "ms\n";
	}

	return 0;
}

	// My cmdline-arguments: -f C:\Users\Patrik\Documents\GitHub\data\firstrun\theRealWorld_png2.lst


/* TODOs:
Hmm, was hältst du von einer FileImageSource und FileListImageSource ?
Bin grad am überlegen, wie ichs unter einen Hut krieg, sowohl von Ordnern, Bildern wie auch einer Bilder-Liste laden zu können
[15:30:44] Patrik: Das nächste problem wird dann, dass ich die Bilder gerne auch mit Landmarks (also gelabelter groundtruth) laden möchte, manchmal, falls vorhanden.
Frage mich grad wie wir das mit den *ImageSource's kombinieren könnten.
[16:10:44] ex-ratt: jo, sowas schwirrte mir auch mal im kopf herum - ground-truth-daten mitladen
[16:11:32] ex-ratt: habe mir aber bisher keine weiteren gedanken gemacht, wollte das aber irgendwie in die image-sources mit reinkriegen bzw. spezielle abgeleitete image-sources basteln, die diese zusatzinfo beinhalten
[16:12:44] Patrik: Jop... ok joa das klingt sehr gut, falls du dich da nicht in den nächsten tagen dran machst, werd ich es tun!

Nochn comment zum oberen: Evtl sollten wir die DirectoryImageSource erweitern, dass sie nur images in der liste hält, die von opencv geladen werden, oft hat man in datenbanken-bilder-dirs auch readme's, oder (Hallo Windows!) Thumbs.db files.
[16:13:07] ex-ratt: jo, da gibts bestimmt file-filters oder sowas
[16:13:22] ex-ratt: bastel erstmal was, ich schau dann mal drüber
[16:13:28] Patrik: Ok :)
[16:14:32] ex-ratt: es gibt eine FaceBoxLandmark
[16:15:00] ex-ratt: aber ich könnte mir vorstellen, dass sowas auch für nicht-gesichter sinn macht
[16:15:23] Patrik: Genau, siehe die klasse Landmark. die FaceBoxLandmark war dann ein wenig "gehacke", weil ich da auch die breite brauch..
[16:16:50] Patrik: Hm, evtl schmeissen wir width/height in die Landmark-klasse, und setzen das einfach 0 wenns nicht benötigt wird, dann fällt ne extra klasse für die face-box weg
*/