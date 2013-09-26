/*
 * faceRecogTestApp.cpp
 *
 *  Created on: 22.05.2013
 *      Author: Patrik Huber
 */

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>

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

#include "faceRecogTestApp.hpp"

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

#ifdef WIN32
#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/iterator/indirect_iterator.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <QtSql>

#include <iostream>
#include <fstream>
#include <chrono>
#include <unordered_map>

//#include "statismo/StatisticalModel.h"
//#include "Representers/VTK/vtkStandardMeshRepresenter.h"

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

bool subjectsMatch(string subject1, string subject2) {
	return subject1.substr(0, 2) == subject2.substr(0, 2);
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(3759128);
	#endif
		
	try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
        ;

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).
                  options(desc).run(), vm);
        po::notify(vm);
    
        if (vm.count("help")) {
            cout << "[faceRecogTestApp] Usage: options_description [options]\n";
            cout << desc;
            return 0;
        }

    }
    catch(std::exception& e) {
        cout << e.what() << "\n";
        return 1;
    }

	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(loglevel::TRACE));

	path fvsdkBins = "C:\\Users\\Patrik\\Cloud\\PhD\\FVSDK_bins\\";
	path firOutDir = "C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\FIRs\\";
	path scoreOutDir = "C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\scores\\";
	path galleryFirList = "C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\mpie_frontal_gallery_fullpath_firlist.lst";

	shared_ptr<FileListImageSource> galleryImageSource = make_shared<FileListImageSource>("C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\mpie_frontal_gallery_fullpath.lst");
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
	}

	// create all FIR's of probes
	
	while (probeImageSource->next()) {
		img = probeImageSource->getImage();
		cmd = fvsdkBins.string() + "enroll.exe " + "-cfg C:\\FVSDK_8_7_0\\etc\\frsdk.cfg " + "-fir " + firOutDir.string() + probeImageSource->getName().stem().string() + ".fir " + "-imgs " + probeImageSource->getName().string();
		int cmdRet = system(cmd.c_str());
		cout << cmdRet;
	}
	*/

	// create the scores of each probe against the whole gallery
	/*
	while (probeImageSource->next()) {
		path probeFirFilepath = path(firOutDir.string() + probeImageSource->getName().stem().string() + ".fir ");
		if (boost::filesystem::exists(probeFirFilepath)) { // We were able to enroll a probe, so there is a FIR, go and match it against gallery
			cmd = fvsdkBins.string() + "match.exe " + "-cfg C:\\FVSDK_8_7_0\\etc\\frsdk.cfg " + "-probe " + probeFirFilepath.string() + "-gallery " + galleryFirList.string() + " -out " + scoreOutDir.string() + probeImageSource->getName().stem().string() + ".txt";
			int cmdRet = system(cmd.c_str());
		} else { // We were not able to enroll the probe - create an empty .txt with content "fte".
			string scoreOutPath = scoreOutDir.string() + probeImageSource->getName().stem().string() + ".txt";
			ofstream myfile;
			myfile.open(scoreOutPath);
			myfile << "FTE" << endl;
			myfile.close();
		}
	}
	*/

	// Go through each probe and change the "FTE"-files to contain all 0.0's with "FTE"-flag
	/*
	while (probeImageSource->next()) {
		path probeFirFilepath = path(firOutDir.string() + probeImageSource->getName().stem().string() + ".fir ");
		if (boost::filesystem::exists(probeFirFilepath)) { // We were able to enroll a probe, so there is a FIR, go and match it against gallery
			//cmd = fvsdkBins.string() + "match.exe " + "-cfg C:\\FVSDK_8_7_0\\etc\\frsdk.cfg " + "-probe " + probeFirFilepath.string() + "-gallery " + galleryFirList.string() + " -out " + scoreOutDir.string() + probeImageSource->getName().stem().string() + ".txt";
			//int cmdRet = system(cmd.c_str());
		} else { // We were not able to enroll the probe - create an empty .txt with content "fte".
			string scoreOutPath = scoreOutDir.string() + probeImageSource->getName().stem().string() + ".txt";
			ofstream scoresFile;
			scoresFile.open(scoreOutPath);

			while (galleryImageSource->next()) {
				string galleryFilename = galleryImageSource->getName().stem().string();
				scoresFile << galleryFilename+".fir" << " " << "0.0" << " " << "FTE" << endl; // Should not contain .fir but the canonical image name
			}
			galleryImageSource->reset();
			scoresFile.close();
			
		}
	}
	*/

	return 0;
}
