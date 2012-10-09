// ffpDetectAppERT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

namespace po = boost::program_options;
using namespace std;

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
    return os;
}

//int _tmain(int argc, _TCHAR* argv[])	// VS10
int main(int argc, char *argv[])		// Peter
{
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(17534);
	
	int verbose_level_text;
	int verbose_level_images;
    bool useFileList = false;
	bool useImgs = false;
	std::string fn_fileList;
	std::vector<std::string> filenames; // Create vector to hold the filenames
	std::vector<Rect> rects;			// Create vector to hold the groundtruth (if there is any)
	
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
            cout << "[ffpDetectAppERT] Usage: options_description [options]\n";
            cout << desc;
            return 0;
        }
        if (vm.count("file-list"))
        {
            cout << "[ffpDetectAppERT] Using file-list as input: " << vm["file-list"].as< string >() << "\n";
			useFileList = true;
			fn_fileList = vm["file-list"].as<string>();
        }
        if (vm.count("input-file"))
        {
            cout << "[ffpDetectAppERT] Using input images: " << vm["input-file"].as<vector<string>>() << "\n";
			useImgs = true;
			filenames = vm["input-file"].as<vector<string>>();
        }
        if (vm.count("verbose-text")) {
            cout << "[ffpDetectAppERT] Verbose level for text: " << vm["verbose-text"].as<int>() << "\n";
        }
        if (vm.count("verbose-images")) {
            cout << "[ffpDetectAppERT] Verbose level for images: " << vm["verbose-images"].as<int>() << "\n";
        }
    }
    catch(std::exception& e) {
        cout << e.what() << "\n";
        return 1;
    }
	if(useFileList==true && useImgs==true) {
		cout << "[ffpDetectAppERT] Error: Please either specify a file-list OR an input-file, not both!" << endl;
		return 1;
	} else if(useFileList==false && useImgs==false) {
		cout << "[ffpDetectAppERT] Error: Please either specify a file-list or an input-file to run the program!" << endl;
		return 1;
	}

	char* fn_detFrontal = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_fd.mat";
	char* fn_detRight= "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_re.mat";	// wvmr	
	char* fn_detLeft = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_le.mat";	// wvml fd_hq64-scasferetmp+40 = subject looking LEFT
	char* fn_regrSVR = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_ra.mat";
	char* fn_regrWVR = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_la.mat";
	
	FdImage *myimg;

	const float DETECT_MAX_DIST_X = 0.33f;
	const float DETECT_MAX_DIST_Y = 0.33f;
	const float DETECT_MAX_DIFF_W = 0.33f;

	//std::string fn_fileList = "D:\\CloudStation\\libFD_patrik2011\\data\\firstrun\\theRealWorld_png.lst";
	//std::string fn_fileList = "H:\\featuredetection\\data\\lfw\\lfw-b_NAS.lst";
	//std::string fn_fileList = "H:\\featuredetection\\data\\feret_m2\\feret-frontal_m2_NAS.lst";
	
	if(useFileList) {
		std::ifstream fileList;
		fileList.open(fn_fileList, std::ios::in);
		if (!fileList.is_open()) {
			std::cout << "[ffpDetectAppERT] Error opening file list!" << std::endl;
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
			rects.push_back(Rect(l, t, r, b));	// GT available or 0 0 0 0
			buf.clear();
			l=t=r=b=0;
		}
		fileList.close();
	}
	// Else useImgs==true: filesnames are already in "filenames", and no groundtruth available!

	// All filesnames now in "filenames", either way
	// All groundtruth now in "rects" IF available (read from .lst)

	Logger->setVerboseLevelText(verbose_level_text);
	Logger->setVerboseLevelImages(verbose_level_images);
	Logger->global.img.writeDetectorCandidates = true;	// Write images of all 5 stages
	Logger->global.img.writeRegressor = true;
	Logger->global.img.writeRegressorPyramids = true;
	Logger->global.img.drawRegressorFoutAsText = true;
	Logger->global.img.drawScales = true;

	int TOT = 0;
	int TACC = 0;
	int FACC = 0;
	int NOCAND = 0;
	int DONTKNOW = 0;

	CascadeERT *ert = new CascadeERT();
	ert->wvm->load(fn_detFrontal);
	ert->svm->load(fn_detFrontal);
	ert->oe->load(fn_detFrontal);
	ert->wvmr->load(fn_detRight);
	ert->svmr->load(fn_detRight);
	ert->oer->load(fn_detRight);
	ert->wvml->load(fn_detLeft);
	ert->svml->load(fn_detLeft);
	ert->oel->load(fn_detLeft);
	ert->wvr->load(fn_regrWVR);
	ert->svr->load(fn_regrSVR);


	for(unsigned int i=0; i< filenames.size(); i++) {
		myimg = new FdImage();
		myimg->load(filenames[i]);

		/* RUN THE WHOLE TREE */
		ert->init_for_image(myimg);
		ert->detect_on_image(myimg);	// See file CascadeERT.cpp for details!
		/* END TREE */


		TOT++;
		if(ert->candidates.size()<1) {
			std::cout << "[ffpDetectAppERT] No face-candidates at all found:  " << filenames[i] << std::endl;
			NOCAND++;
		} else {
			if(rects[i]==Rect(0, 0, 0, 0)) {//no groundtruth
				std::cout << "[ffpDetectAppERT] No ground-truth available, not counting anything: " << filenames[i] << std::endl;
				++DONTKNOW;
			} else {//we have groundtruth
				int gt_w = abs(rects[i].right-rects[i].left);
				int gt_h = abs(rects[i].top-rects[i].bottom);
				int gt_cx = rects[i].left+gt_w/2;
				int gt_cy = rects[i].top+gt_h/2;
				if (abs(gt_cx - ert->candidates[0]->c.x) < DETECT_MAX_DIST_X*(float)gt_w &&
					abs(gt_cy - ert->candidates[0]->c.y) < DETECT_MAX_DIST_Y*(float)gt_w &&
					abs(gt_w - ert->candidates[0]->w_inFullImg) < DETECT_MAX_DIFF_W*(float)gt_w       ) {
				
					std::cout << "[ffpDetectAppERT] TACC (1/1): " << filenames[i] << std::endl;
					TACC++;
				} else {
					std::cout << "[ffpDetectAppERT] Face not found, wrong position:  " << filenames[i] << std::endl;
					FACC++;
				}
			}//end no groundtruth
		}

		std::cout << std::endl;
		std::cout << "[ffpDetectAppERT] -------------------------------------" << std::endl;
		std::cout << "[ffpDetectAppERT] TOT:  " << TOT << std::endl;
		std::cout << "[ffpDetectAppERT] TACC:  " << TACC << std::endl;
		std::cout << "[ffpDetectAppERT] FACC:  " << FACC << std::endl;
		std::cout << "[ffpDetectAppERT] NOCAND:  " << NOCAND << std::endl;
		std::cout << "[ffpDetectAppERT] DONTKNOW:  " << DONTKNOW << std::endl;
		std::cout << "[ffpDetectAppERT] -------------------------------------" << std::endl;

		delete myimg;
	}

	std::cout << std::endl;
	std::cout << "[ffpDetectAppERT] =====================================" << std::endl;
	std::cout << "[ffpDetectAppERT] =====================================" << std::endl;
	std::cout << "[ffpDetectAppERT] TOT:  " << TOT << std::endl;
	std::cout << "[ffpDetectAppERT] TACC:  " << TACC << std::endl;
	std::cout << "[ffpDetectAppERT] FACC:  " << FACC << std::endl;
	std::cout << "[ffpDetectAppERT] NOCAND:  " << NOCAND << std::endl;
	std::cout << "[ffpDetectAppERT] DONTKNOW:  " << DONTKNOW << std::endl;
	std::cout << "[ffpDetectAppERT] =====================================" << std::endl;

	delete ert;

	return 0;
}

