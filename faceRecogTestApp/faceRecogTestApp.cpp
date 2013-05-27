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

#include <frsdk/config.h>
#include <frsdk/face.h>

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
		cout << "[faceRecogTestApp] Error: Please either specify a file-list, an input-file or a directory (and only one of them) to run the program!" << endl;
		return 1;
	}

	string cfgFn = "C:\\FVSDK_8_7_0\\etc\\frsdk.cfg";
	FRsdk::Configuration cfg( cfgFn );

	FRsdk::Face::Finder faceFinder( cfg);

	Mat img;
	while ((img = imageSource->getImage()).rows != 0) { // TODO Can we check against !=Mat() somehow? or .empty?

		cv::namedWindow("src", CV_WINDOW_AUTOSIZE); cv::imshow("src", img);
		cvMoveWindow("src", 0, 0);
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		string filePath = string( imageSource->getName().string() );
		
		

		float minrEyeDist = 0.1f;

		FRsdk::Image imgCog( FRsdk::ImageIO::load( filePath ));

		float mindist = 0.1f;
		float maxdist = 0.4f;

		// doing face finding
		//shared_ptr<FRsdk::Face::LocationSet> locations = make_shared<FRsdk::Face::LocationSet>(			
		//	faceFinder.find (imgCog, mindist, maxdist)
		//	);
		FRsdk::Face::LocationSet* locations = new FRsdk::Face::LocationSet(			
				faceFinder.find (imgCog, mindist, maxdist)
				);

		std::cout << endl << "number of found faces: " << locations->size () << endl;

		FRsdk::Face::LocationSet::const_iterator faceIter = locations->begin();
		while( faceIter != locations->end()) {
			cout << "Face location: [" << (*faceIter).pos.x() << ", " 
				<< (*faceIter).pos.y() << "], width=" 
				<< (*faceIter).width << ", confidence=" 
				<< (*faceIter).confidence << ", rotationAngle="
				<< (*faceIter).rotationAngle 
				<< endl;
			faceIter++;
		}


		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);
		std::cout << "Elapsed time: " << elapsed_mseconds << "ms\n";
		imageSource->next();
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