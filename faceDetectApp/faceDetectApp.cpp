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
	
	ptree pt;
	read_info("C:\\Users\\Patrik\\Documents\\GitHub\\ffpDetectApp.cfg", pt);		
	string wvmClassifierFile = pt.get<string>("detection.wvm.classifierFile");
	string wvmThresholdsFile = pt.get<string>("detection.wvm.thresholdsFile");
	string svmClassifierFile = pt.get<string>("detection.svm.classifierFile");
	string svmLogisticParametersFile = pt.get<string>("detection.svm.logisticParametersFile");

	float svmThreshold = pt.get<float>("detection.svm.threshold", -1.2f);	// If the key doesn't exist in the config, set it to -1.2 (default value).

	/* Note: We could change/write/add something to the config with
	pt.put("detection.svm.threshold", -0.5f);
	If the value already exists, it gets overwritten, if not, it gets created.
	Save it with:
	write_info("C:\\Users\\Patrik\\Documents\\GitHub\\ffpDetectApp.cfg", pt);
	*/

	shared_ptr<ProbabilisticWvmClassifier> pwvm = ProbabilisticWvmClassifier::loadMatlab(wvmClassifierFile, wvmThresholdsFile);
	shared_ptr<ProbabilisticSvmClassifier> psvm = ProbabilisticSvmClassifier::loadMatlab(svmClassifierFile, svmLogisticParametersFile);

	shared_ptr<ImagePyramid> pyr = make_shared<ImagePyramid>(pt.get<float>("detection.imagePyramid.minScaleFactor", 0.09f), pt.get<float>("detection.imagePyramid.maxScaleFactor", 0.25f), pt.get<float>("detection.imagePyramid.incrementalScaleFactor", 0.9f));	// (0.09, 0.25, 0.9) is nearly the same as old 90, 9, 0.9
	pyr->addImageFilter(make_shared<GrayscaleFilter>());
	shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(pyr, 20, 20);
	featureExtractor->addPatchFilter(make_shared<HistEq64Filter>());

	shared_ptr<SlidingWindowDetector> det = make_shared<SlidingWindowDetector>(pwvm, featureExtractor);
	shared_ptr<OverlapElimination> oe = make_shared<OverlapElimination>(pt.get<float>("detection.overlapElimination.dist", 5.0f), pt.get<float>("detection.overlapElimination.ratio", 0.0f));
	psvm->getSvm()->setThreshold(svmThreshold);

	Mat img;

	for(unsigned int i=0; i< filenames.size(); i++) {
		img = cv::imread(filenames[i]);
		cv::namedWindow("src", CV_WINDOW_AUTOSIZE); cv::imshow("src", img);
		cvMoveWindow("src", 0, 0);
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();
		vector<shared_ptr<ClassifiedPatch>> resultingPatches = det->detect(img);

		Mat imgWvm = img.clone();
		for(auto pit = resultingPatches.begin(); pit != resultingPatches.end(); pit++) {
			shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
			shared_ptr<Patch> patch = classifiedPatch->getPatch();
			cv::rectangle(imgWvm, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
		}
		cv::namedWindow("wvm", CV_WINDOW_AUTOSIZE); cv::imshow("wvm", imgWvm);
		cvMoveWindow("wvm", 0, 0);
	}

	return 0;
}

	// My cmdline-arguments: -f C:\Users\Patrik\Documents\GitHub\data\firstrun\theRealWorld_png2.lst
