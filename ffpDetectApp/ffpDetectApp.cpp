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
	#include <crtdbg.h>
#endif

#ifdef _DEBUG
	#ifndef DBG_NEW
		#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
		#define new DBG_NEW
	#endif
#endif  // _DEBUG

#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/iterator/indirect_iterator.hpp"

#include "classification/RbfKernel.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/WvmClassifier.hpp"
#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/PyramidFeatureExtractor.hpp"
#include "imageprocessing/HistEq64Filter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"

#include "detection/SlidingWindowDetector.hpp"
#include "detection/ClassifiedPatch.hpp"

#include "logging/LoggerFactory.hpp"


namespace po = boost::program_options;
using namespace std;
using namespace imageprocessing;
using namespace detection;
using namespace classification;
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;
using boost::make_indirect_iterator;

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
	std::string fn_fileList;
	std::vector<std::string> filenames; // Create vector to hold the filenames
	std::vector<cv::Rect> rects;			// Create vector to hold the groundtruth (if there is any).
										// Format (old): l t r b. OpenCV: ??
	
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

	const float DETECT_MAX_DIST_X = 0.33f;
	const float DETECT_MAX_DIST_Y = 0.33f;
	const float DETECT_MAX_DIFF_W = 0.33f;

	//std::string fn_fileList = "D:\\CloudStation\\libFD_patrik2011\\data\\firstrun\\theRealWorld_png.lst";
	//std::string fn_fileList = "H:\\featuredetection\\data\\lfw\\lfw-b_NAS.lst";
	//std::string fn_fileList = "H:\\featuredetection\\data\\feret_m2\\feret-frontal_m2_NAS.lst";
	
	if(useFileList) {
		std::ifstream fileList;
		fileList.open(fn_fileList.c_str(), std::ios::in);
		if (!fileList.is_open()) {
			std::cout << "[ffpDetectApp] Error opening file list!" << std::endl;
			return 0;
		}
		std::string line;
		while( fileList.good() ) {
			std::getline(fileList, line);
			if(line=="") {
				continue;
			}
      		std::string buf; // Have a buffer string
			int l=0, r=0, b=0, t=0;
			std::stringstream ss(line); // Insert the string into a stream
			//while (ss >> buf)
			ss >> buf;	
			filenames.push_back(buf);
			ss >> l;
			ss >> t;
			ss >> r;
			ss >> b;
			//if(!(l==0 && t==0 && r==0 && b==0))
			rects.push_back(cv::Rect(l, t, r-l, b-t));	// GT available or 0 0 0 0
			buf.clear();
			l=t=r=b=0;
		}
		fileList.close();
	}
	// Else useImgs==true: filesnames are already in "filenames", and no groundtruth available!

	// All filesnames now in "filenames", either way
	// All groundtruth now in "rects" IF available (read from .lst)

	int TOT = 0;
	int TACC = 0;
	int FACC = 0;
	int NOCAND = 0;
	int DONTKNOW = 0;

	//Logger->setVerboseLevelImages(verbose_level_images);
	//Logger->global.img.writeDetectorCandidates = true;	// Write images of all 5 stages

	shared_ptr<ProbabilisticWvmClassifier> pwvm = ProbabilisticWvmClassifier::loadMatlab("C:/Users/Patrik/Documents/GitHub/config/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--With-outnew02-HQ64SVM.mat", "C:/Users/Patrik/Documents/GitHub/config/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--ts107742-hq64_thres_0.001--with-outnew02HQ64SVM.mat");

	shared_ptr<ImagePyramid> pyr = make_shared<ImagePyramid>(0.09, 0.25, 0.9);	// (0.09, 0.25, 0.9) is nearly the same as old 90, 9, 0.9
	pyr->addImageFilter(make_shared<GrayscaleFilter>());
	shared_ptr<PyramidFeatureExtractor> featureExtractor = make_shared<PyramidFeatureExtractor>(pyr, 20, 20);
	featureExtractor->addPatchFilter(make_shared<HistEq64Filter>());

	shared_ptr<SlidingWindowDetector> det = make_shared<SlidingWindowDetector>(pwvm, featureExtractor);

	Mat img;
	for(unsigned int i=0; i< filenames.size(); i++) {
		img = cv::imread(filenames[i]);
		cv::namedWindow("src", CV_WINDOW_AUTOSIZE); cv::imshow("src", img);
		// update pyr etc...? done in detector?
		vector<shared_ptr<ClassifiedPatch>> resultingPatches = det->detect(img);
		//Logger->LogImgDetectorFinal(myimg, casc->candidates, casc->svm->getIdentifier(), "Final");
		Mat rgbimg = img.clone();
		for(auto pit = resultingPatches.begin(); pit != resultingPatches.end(); pit++) {
			shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
			shared_ptr<Patch> patch = classifiedPatch->getPatch();
			cv::rectangle(rgbimg, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
		}
		cv::namedWindow("final", CV_WINDOW_AUTOSIZE); cv::imshow("final", rgbimg);
		
		sort(make_indirect_iterator(resultingPatches.begin()), make_indirect_iterator(resultingPatches.end()), greater<ClassifiedPatch>());
		Mat finalimg = img.clone();
		cv::rectangle(finalimg, cv::Point(resultingPatches[0]->getPatch()->getX() - resultingPatches[0]->getPatch()->getWidth()/2, resultingPatches[0]->getPatch()->getY() - resultingPatches[0]->getPatch()->getHeight()/2), cv::Point(resultingPatches[0]->getPatch()->getX() + resultingPatches[0]->getPatch()->getWidth()/2, resultingPatches[0]->getPatch()->getY() + resultingPatches[0]->getPatch()->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((resultingPatches[0]->getProbability())/1.0)   ));
		cv::namedWindow("final1", CV_WINDOW_AUTOSIZE); cv::imshow("final1", finalimg);
		cv::waitKey();

		TOT++;
		if(resultingPatches.size()<1) {
			std::cout << "[ffpDetectApp] No face-candidates at all found:  " << filenames[i] << std::endl;
			NOCAND++;
		} else {
			if(rects[i]==Rect(0, 0, 0, 0)) {//no groundtruth
				std::cout << "[ffpDetectApp] No ground-truth available, not counting anything: " << filenames[i] << std::endl;
				++DONTKNOW;
			} else {//we have groundtruth
				int gt_w = rects[i].width;
				int gt_h = rects[i].height;
				int gt_cx = rects[i].x+gt_w/2;
				int gt_cy = rects[i].y+gt_h/2;
				if (abs(gt_cx - resultingPatches[0]->getPatch()->getX()) < DETECT_MAX_DIST_X*(float)gt_w &&
					abs(gt_cy - resultingPatches[0]->getPatch()->getY()) < DETECT_MAX_DIST_Y*(float)gt_w &&
					abs(gt_w - resultingPatches[0]->getPatch()->getWidth()) < DETECT_MAX_DIFF_W*(float)gt_w       ) {
				
					std::cout << "[ffpDetectApp] TACC (1/1): " << filenames[i] << std::endl;
					TACC++;
				} else {
					std::cout << "[ffpDetectApp] Face not found, wrong position:  " << filenames[i] << std::endl;
					FACC++;
				}
			}//end no groundtruth
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
