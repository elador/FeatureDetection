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
#include "imageio/FaceBoxLandmark.hpp"

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
				if (patch->getProbability() > faceRegionProbabilityMap.at<float>(currY, currX)) {
					faceRegionProbabilityMap.at<float>(currY, currX) = patch->getProbability();
				}
			}
		}
	}

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
	string fn_fileList;
	vector<std::string> filenames;
	vector<FaceBoxLandmark> groundtruthFaceBoxes;
	
	try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("verbose-text,v", po::value<int>(&verbose_level_text)->implicit_value(2)->default_value(1,"minimal text output"),
                  "enable text-verbosity (optionally specify level)")
            ("verbose-images,w", po::value<int>(&verbose_level_images)->implicit_value(2)->default_value(1,"minimal image output"),
                  "enable image-verbosity (optionally specify level)")
            ("file-list,f", po::value< string >(), 
                  "a .lst file to process")
            ("input-file,i", po::value< vector<string> >(), "input image")
        ;

        po::positional_options_description p;
        p.add("input-file", -1);
        
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).
                  options(desc).positional(p).run(), vm);
        po::notify(vm);
    
        if (vm.count("help")) {
            cout << "[ffpDetectApp] Usage: options_description [options]\n";
            cout << desc;
            return 0;
        }
        if (vm.count("file-list"))
        {
            cout << "[ffpDetectApp] Using file-list as input: " << vm["file-list"].as< string >() << "\n";
			useFileList = true;
			fn_fileList = vm["file-list"].as<string>();
        }
        if (vm.count("input-file"))
        {
            cout << "[ffpDetectApp] Using input images: " << vm["input-file"].as< vector<string> >() << "\n";
			useImgs = true;
			filenames = vm["input-file"].as< vector<string> >();
        }
        if (vm.count("verbose-text")) {
            cout << "[ffpDetectApp] Verbose level for text: " << vm["verbose-text"].as<int>() << "\n";
        }
        if (vm.count("verbose-images")) {
            cout << "[ffpDetectApp] Verbose level for images: " << vm["verbose-images"].as<int>() << "\n";
        }
    }
    catch(std::exception& e) {
        cout << e.what() << "\n";
        return 1;
    }
	if(useFileList==true && useImgs==true) {
		cout << "[ffpDetectApp] Error: Please either specify a file-list OR an input-file, not both!" << endl;
		return 1;
	} else if(useFileList==false && useImgs==false) {
		cout << "[ffpDetectApp] Error: Please either specify a file-list or an input-file to run the program!" << endl;
		return 1;
	}

	if(useFileList) {
		std::ifstream fileList;
		fileList.open(fn_fileList.c_str(), std::ios::in);
		if (!fileList.is_open()) {
			std::cout << "[ffpDetectApp] Error opening file list!" << std::endl;
			return 0;
		}
		string line;
		while(fileList.good()) {
			getline(fileList, line);
			if(line=="") {
				continue;
			}
			string buf;
			stringstream ss(line);
			ss >> buf;	
			filenames.push_back(buf);	// Insert the image filename
			groundtruthFaceBoxes.push_back(LandmarksHelper::readFromLstLine(line));	// Insert the groundtruth facebox
		}
		fileList.close();
	}
	// Else useImgs==true: filesnames are already in "filenames", and no groundtruth available!

	/* Testing ground */

	/* END */

	Loggers->getLogger("classification").addAppender(make_shared<logging::ConsoleAppender>(loglevel::TRACE));
	
	ptree pt;
	read_info("C:\\Users\\Patrik\\Documents\\GitHub\\faceDetectApp.cfg", pt);		

	ptree ptClassifiers = pt.get_child("classifiers");
	unordered_multimap<string, shared_ptr<ProbabilisticClassifier>> classifiers;
	for_each(begin(ptClassifiers), end(ptClassifiers), [&classifiers](ptree::value_type kv) {
		// TODO: 1) error handling if a key doesn't exist
		//		 2) make more dynamic, search for wvm or svm and add respective to map
		string classifierFile = kv.second.get<string>("wvm.classifierFile");
		string thresholdsFile = kv.second.get<string>("wvm.thresholdsFile");
		classifiers.insert(make_pair(kv.first + ".wvm", ProbabilisticWvmClassifier::loadMatlab(classifierFile, thresholdsFile)));		
	});

	shared_ptr<ImagePyramid> pyr = make_shared<ImagePyramid>(pt.get<float>("imagePyramid.minScaleFactor", 0.09f), pt.get<float>("imagePyramid.maxScaleFactor", 0.25f), pt.get<float>("imagePyramid.incrementalScaleFactor", 0.9f));	// (0.09, 0.25, 0.9) is nearly the same as old 90, 9, 0.9
	pyr->addImageFilter(make_shared<GrayscaleFilter>());
	shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(pyr, 20, 20);
	featureExtractor->addPatchFilter(make_shared<HistEq64Filter>());

	// TODO: Better: include this in the loading above. e.g. create ProbabilisticWvmClassifier::load(ptree), which
	// loads ProbabilisticWvmClassifier::loadMatlab and sets the parameters.
	shared_ptr<ProbabilisticWvmClassifier> test = dynamic_pointer_cast<ProbabilisticWvmClassifier>(classifiers.find("faceFrontal.wvm")->second);
	test->getWvm()->setNumUsedFilters(280);

	unordered_multimap<string, shared_ptr<SlidingWindowDetector>> detectors;
	for(auto classifier : classifiers) {
		detectors.insert(make_pair(classifier.first, make_shared<SlidingWindowDetector>(classifier.second, featureExtractor)));
	}

	Mat img;

	for(unsigned int i=0; i< filenames.size(); i++) {
		img = cv::imread(filenames[i]);
		cv::namedWindow("src", CV_WINDOW_AUTOSIZE); cv::imshow("src", img);
		cvMoveWindow("src", 0, 0);
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		int det=0;
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

			

			/*Mat faceRegionProbabilityMap = getFaceRegionProbabilityMapFromPatchlist(resultingPatches, img.cols, img.rows);
			faceRegionProbabilityMap *= 255.0f;
			faceRegionProbabilityMap.convertTo(faceRegionProbabilityMap, CV_8UC1);
			cv::namedWindow("pr", CV_WINDOW_AUTOSIZE); cv::imshow("pr", faceRegionProbabilityMap);
			cvMoveWindow("pr", 512, 0);
			cv::waitKey(0);
			*/
			++det;
		}
		cv::waitKey(0);
		
		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);
		std::cout << "Elapsed time: " << elapsed_mseconds << "ms\n";
	}

	return 0;
}

	// My cmdline-arguments: -f C:\Users\Patrik\Documents\GitHub\data\firstrun\theRealWorld_png2.lst
