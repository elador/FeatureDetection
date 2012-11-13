// ffpDetectApp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "CascadeFacialFeaturePoints.h"

#include "SLogger.h"
#include "FdImage.h"

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

	//std::string fn_detFrontal = "C:\\Users\\Patrik\\Documents\\GitHub\\config\\cfg\\faceDetectApp\\fd_config_ffd_fd.mat";
	//std::string fn_detFrontal = "C:\\Users\\Patrik\\Documents\\GitHub\\config\\fdetection\\fd_config_ffd_le.mat";
	
	FdImage *myimg;
	
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
			std::stringstream ss(line); // Insert the string into a stream
			ss >> buf;	
			filenames.push_back(buf);
			buf.clear();
		}
		fileList.close();
	}
	// Else useImgs==true: filesnames are already in "filenames", and no groundtruth available!


	Logger->setVerboseLevelText(verbose_level_text);
	Logger->setVerboseLevelImages(verbose_level_images);
	Logger->global.img.writeDetectorCandidates = true;	// Write images of all 5 stages
	Logger->global.img.drawScales = true;
	Logger->global.img.writeImgPyramids = true;

	/* Testing ground */

	/* END */
	
	CascadeFacialFeaturePoints* casc = new CascadeFacialFeaturePoints();
	casc->setIdentifier("smartFaceDetect");

	for(unsigned int i=0; i< filenames.size(); i++) {
		myimg = new FdImage();
		myimg->load(filenames[i]);
		casc->initForImage(myimg);
		casc->detectOnImage(myimg);
		//Logger->LogImgDetectorFinal(myimg, casc->candidates, casc->svm->getIdentifier(), "Final");

		delete myimg;
	}

	delete casc;

	return 0;
}

	// My cmdline-arguments: -f C:\Users\Patrik\Documents\GitHub\data\firstrun\theRealWorld_png2.lst
