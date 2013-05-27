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
#include "imageio/Landmark.hpp"

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
	//_CrtSetBreakAlloc(287);
	#endif
	
	int verbose_level_text;
	int verbose_level_images;
    bool useFileList = false;
	bool useImgs = false;
	string fn_fileList;
	string configFilename;
	vector<std::string> filenames;
	vector<Landmark> groundtruthFaceBoxes;
	
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
			("config,c", po::value< string >(), 
				  "path to a config (.cfg) file")
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
		if (vm.count("config"))
		{
			cout << "[ffpDetectApp] Using config: " << vm["config"].as< string >() << "\n";
			configFilename = vm["config"].as< string >();
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

	const float DETECT_MAX_DIST_X = 0.33f;	// --> Config
	const float DETECT_MAX_DIST_Y = 0.33f;
	const float DETECT_MAX_DIFF_W = 0.33f;

	if(useFileList) {
		ifstream fileList;
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
			//TODO CURRENTLY BROKEN groundtruthFaceBoxes.push_back(LandmarksHelper::readFromLstLine(line));	// Insert the groundtruth facebox
		}
		fileList.close();
	}
	// Else useImgs==true: filesnames are already in "filenames", and no groundtruth available!

	// All filesnames now in "filenames", either way
	// All groundtruth now in "groundtruthFaceBoxes" IF available (read from .lst)

	int TOT = 0;
	int TACC = 0;
	int FACC = 0;
	int NOCAND = 0;
	int DONTKNOW = 0;

	ptree pt;
	read_info(configFilename, pt);		
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

		img = cv::imread(filenames[i]);		// TODO: Add a check if the file really exists. But we should use ImageSource soon anyway.
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

		resultingPatches = oe->eliminate(resultingPatches);
		Mat imgWvmOe = img.clone();
		for(auto pit = resultingPatches.begin(); pit != resultingPatches.end(); pit++) {
			shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
			shared_ptr<Patch> patch = classifiedPatch->getPatch();
			cv::rectangle(imgWvmOe, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
		}
		cv::namedWindow("wvmoe", CV_WINDOW_AUTOSIZE); cv::imshow("wvmoe", imgWvmOe);
		cvMoveWindow("wvmoe", 550, 0);

		vector<shared_ptr<ClassifiedPatch>> svmPatches;
		for(auto &patch : resultingPatches) {
			svmPatches.push_back(make_shared<ClassifiedPatch>(patch->getPatch(), psvm->classify(patch->getPatch()->getData())));
		}

		vector<shared_ptr<ClassifiedPatch>> svmPatchesPositive;
		Mat imgSvm = img.clone();
		for(auto pit = svmPatches.begin(); pit != svmPatches.end(); pit++) {
			shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
			shared_ptr<Patch> patch = classifiedPatch->getPatch();
			if(classifiedPatch->isPositive()) {
				cv::rectangle(imgSvm, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
				svmPatchesPositive.push_back(classifiedPatch);
			}
		}
		cv::namedWindow("svm", CV_WINDOW_AUTOSIZE); cv::imshow("svm", imgSvm);
		cvMoveWindow("svm", 0, 500);

		sort(make_indirect_iterator(svmPatches.begin()), make_indirect_iterator(svmPatches.end()), greater<ClassifiedPatch>());

		Mat imgSvmEnd = img.clone();
		cv::rectangle(imgSvmEnd, cv::Point(svmPatches[0]->getPatch()->getX() - svmPatches[0]->getPatch()->getWidth()/2, svmPatches[0]->getPatch()->getY() - svmPatches[0]->getPatch()->getHeight()/2), cv::Point(svmPatches[0]->getPatch()->getX() + svmPatches[0]->getPatch()->getWidth()/2, svmPatches[0]->getPatch()->getY() + svmPatches[0]->getPatch()->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((svmPatches[0]->getProbability())/1.0)   ));
		cv::namedWindow("svmEnd", CV_WINDOW_AUTOSIZE); cv::imshow("svmEnd", imgSvmEnd);
		cvMoveWindow("svmEnd", 550, 500);

		end = std::chrono::system_clock::now();

		int elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);

		std::cout << "finished computation at " << std::ctime(&end_time)
			<< "elapsed time: " << elapsed_seconds << "s, " << elapsed_mseconds << "ms\n";

		cv::waitKey(30);

		TOT++;
		if(resultingPatches.size()<1) {
			std::cout << "[ffpDetectApp] No face-candidates at all found:  " << filenames[i] << std::endl;
			NOCAND++;
		} else {
			// TODO Check if the LM exists or it will crash! Currently broken!
			if(groundtruthFaceBoxes[i].getPosition3D()==Vec3f(0.0f, 0.0f, 0.0f)) { //no groundtruth
				std::cout << "[ffpDetectApp] No ground-truth available, not counting anything: " << filenames[i] << std::endl;
				++DONTKNOW;
			} else { //we have groundtruth
				int gt_w = groundtruthFaceBoxes[i].getWidth();
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
				}
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
