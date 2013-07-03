/*
 * faceRecogTestApp.cpp
 *
 *  Created on: 22.05.2013
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

#include "imageio/Landmark.hpp"
#include "imageio/LandmarksHelper.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/NamedLabeledImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/RepeatingFileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/DidLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

#include "faceRecogTestApp.hpp"

namespace po = boost::program_options;
using namespace std;
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


/*
void drawFaceBoxes(Mat image, vector<shared_ptr<ClassifiedPatch>> patches)
{
	for(auto pit = patches.begin(); pit != patches.end(); pit++) {
		shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
		shared_ptr<Patch> patch = classifiedPatch->getPatch();
		cv::rectangle(image, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
	}
}
*/

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
            cout << "[faceRecogTestApp] Usage: options_description [options]\n";
            cout << desc;
            return 0;
        }
        if (vm.count("input-list"))
        {
            cout << "[faceRecogTestApp] Using file-list as input: " << vm["input-list"].as<string>() << "\n";
			useFileList = true;
			inputFilelist = vm["input-list"].as<string>();
        }
        if (vm.count("input-file"))
        {
            cout << "[faceRecogTestApp] Using input images: " << vm["input-file"].as<vector<string>>() << "\n";
			useImgs = true;
			inputFilenames = vm["input-file"].as< vector<string> >();
        }
		if (vm.count("input-dir"))
		{
			cout << "[faceRecogTestApp] Using input images: " << vm["input-dir"].as<string>() << "\n";
			useDirectory = true;
			inputDirectory = vm["input-dir"].as<string>();
		}
        if (vm.count("verbose-text")) {
            cout << "[faceRecogTestApp] Verbose level for text: " << vm["verbose-text"].as<int>() << "\n";
        }
        if (vm.count("verbose-images")) {
            cout << "[faceRecogTestApp] Verbose level for images: " << vm["verbose-images"].as<int>() << "\n";
        }
    }
    catch(std::exception& e) {
        cout << e.what() << "\n";
        return 1;
    }

	Loggers->getLogger("classification").addAppender(make_shared<logging::ConsoleAppender>(loglevel::TRACE));
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(loglevel::TRACE));
	Loggers->getLogger("detection").addAppender(make_shared<logging::ConsoleAppender>(loglevel::TRACE));

	int numInputs = 0;
	if(useFileList==true) {
		numInputs++;
	}
	if(useImgs==true) {
		numInputs++;
	}
	if(useDirectory==true) {
		numInputs++;
	}
	if(numInputs!=1) {
		cout << "[faceRecogTestApp] Error: Please either specify a file-list, an input-file or a directory (and only one of them) to run the program!" << endl;
		return 1;
	}

	path fvsdkBins = "C:\\Users\\Patrik\\Documents\\GitHub\\FVSDK_bins\\";
	path firOutDir = "C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\FIRs\\";
	path scoreOutDir = "C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\scores\\";
	path galleryFirList = "C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\mpie_frontal_gallery_fullpath_firlist.lst";

	shared_ptr<ImageSource> galleryImageSource = make_shared<FileListImageSource>("C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\mpie_frontal_gallery_fullpath.lst");
	shared_ptr<ImageSource> probeImageSource = make_shared<FileListImageSource>("C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\mpie_probe_fullpath.lst");

	Mat img;
	string cmd;
	// create all FIR's of gallery
	/*
	while (galleryImageSource->next()) {
		img = galleryImageSource->getImage();
		cmd = fvsdkBins.string() + "enroll.exe " + "-cfg C:\\FVSDK_8_7_0\\etc\\frsdk.cfg " + "-fir " + firOutDir.string() + galleryImageSource->getName().stem().string() + ".fir " + "-imgs " + galleryImageSource->getName().string();
		int cmdRet = system(cmd.c_str());
		cout << cmdRet;
	}*/

	// create all FIR's of probes
	/*
	while (probeImageSource->next()) {
		img = probeImageSource->getImage();
		cmd = fvsdkBins.string() + "enroll.exe " + "-cfg C:\\FVSDK_8_7_0\\etc\\frsdk.cfg " + "-fir " + firOutDir.string() + probeImageSource->getName().stem().string() + ".fir " + "-imgs " + probeImageSource->getName().string();
		int cmdRet = system(cmd.c_str());
		cout << cmdRet;
	}*/

	// create the scores of each probe against the whole gallery
	while (probeImageSource->next()) {
		cmd = fvsdkBins.string() + "match.exe " + "-cfg C:\\FVSDK_8_7_0\\etc\\frsdk.cfg " + "-probe " + firOutDir.string() + probeImageSource->getName().stem().string() + ".fir " + "-gallery " + galleryFirList.string() + " -out " + scoreOutDir.string() + probeImageSource->getName().stem().string() + ".fir";
		int cmdRet = system(cmd.c_str());
	}


	return 0;
}
