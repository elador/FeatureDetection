/*
 * frDbExp.cpp
 *
 *  Created on: 13.08.2013
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

#include "frDbExp.hpp"

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
#include <vector>
#include <algorithm>

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

void writeAscii(path filename, vector<float> positiveScores, vector<float> negativeScores, int numFailureToEnroll, string experimentTitle, string curveLabel) {

	ofstream file(filename.string());

	if (file.is_open()) {

		file << experimentTitle << endl;
		file << curveLabel << endl;
		file << "fte " << numFailureToEnroll << endl;
		copy(begin(positiveScores), end(positiveScores), ostream_iterator<float>(file, " "));
		file << endl;
		copy(begin(negativeScores), end(negativeScores), ostream_iterator<float>(file, " "));
		file << endl;

		file.close();

	}
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
            ("help,h", "Produce help message.")
        ;

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).
                  options(desc).run(), vm);
        po::notify(vm);
    
        if (vm.count("help")) {
            cout << "[frDbExp] Usage: frDbExp [options]\n";
            cout << desc;
            return 0;
        }

    }
    catch(std::exception& e) {
        cout << e.what() << "\n";
        return 1;
    }

	//string databaseFilename = "C:\\Users\\Patrik\\Cloud\\PhD\\data\\frdb.sqlite";
	string databaseFilename = "C:\\Users\\Patrik\\frdb.sqlite";

	QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE");
	db.setDatabaseName(QString::fromStdString(databaseFilename));
	if (!db.open())	{
		cout << db.lastError().text().toStdString() << endl;
	}
	QSqlQuery query;
	bool ret;
	ret = query.exec("PRAGMA foreign_keys = ON");
	if (!ret) {
		cout << query.lastError().text().toStdString() << endl;
	}

	vector<float> positiveScores;
	vector<float> negativeScores;
	int numFailureToEnroll = 0;

	query.setForwardOnly(true);
	//ret = query.exec("SELECT s.* FROM scores s INNER JOIN images ip ON s.probe = ip.filepath INNER JOIN images ig ON s.gallery = ig.filepath WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0 OR ip.yaw=-15.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject=ig.subject"); // (if I had more images, add WHERE db=multipie, lighting=..., etc.) // AND (ip.yaw=60.0 OR ip.yaw=45) // AND ip.subject=001
	ret = query.exec("SELECT s.* FROM scores s INNER JOIN images ip ON s.probe = ip.filepath INNER JOIN images ig ON s.gallery = ig.filepath WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0 OR ip.yaw=-15.0 OR ip.yaw=30.0 OR ip.yaw=-30.0 OR ip.yaw=45.0 OR ip.yaw=-45.0 OR ip.yaw=60.0 OR ip.yaw=-60.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject=ig.subject"); // (if I had more images, add WHERE db=multipie, lighting=..., etc.) // AND (ip.yaw=60.0 OR ip.yaw=45) // AND ip.subject=001
	if (!ret) {
		cout << query.lastError().text().toStdString() << endl;
	}
	while (query.next()) {
		if (query.value(3).toInt() == 1) {
			++numFailureToEnroll;
		} else {
			positiveScores.push_back(query.value(2).toFloat());
		}
	}

	//ret = query.exec("SELECT s.* FROM scores s INNER JOIN images ip ON s.probe = ip.filepath INNER JOIN images ig ON s.gallery = ig.filepath WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0 OR ip.yaw=-15.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject!=ig.subject"); // (if I had more images, add WHERE db=multipie, lighting=..., etc.) // AND (ip.yaw=60.0 OR ip.yaw=45) // AND ip.subject=001
	ret = query.exec("SELECT s.* FROM scores s INNER JOIN images ip ON s.probe = ip.filepath INNER JOIN images ig ON s.gallery = ig.filepath WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0 OR ip.yaw=-15.0 OR ip.yaw=30.0 OR ip.yaw=-30.0 OR ip.yaw=45.0 OR ip.yaw=-45.0 OR ip.yaw=60.0 OR ip.yaw=-60.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject!=ig.subject"); // (if I had more images, add WHERE db=multipie, lighting=..., etc.) // AND (ip.yaw=60.0 OR ip.yaw=45) // AND ip.subject=001
	if (!ret) {
		cout << query.lastError().text().toStdString() << endl;
	}
	while (query.next()) {
		if (query.value(3).toInt() != 1) {
			negativeScores.push_back(query.value(2).toFloat());
		}
	}
	
	db.close();
	QSqlDatabase::removeDatabase("QSQLITE");

	sort(begin(positiveScores), end(positiveScores), less<float>()); // 0 to 1
	sort(begin(negativeScores), end(negativeScores), greater<float>()); // 1 to 0

	//writeAscii("C:/Users/Patrik/Documents/MATLAB/probes_pm15_fte.txt", positiveScores, negativeScores, numFailureToEnroll, "Probes yaw = +-15, with FTE's", "yaw=+-15");
	writeAscii("C:/Users/Patrik/Documents/MATLAB/probes_all.txt", positiveScores, negativeScores, numFailureToEnroll, "Probes -60 to +60 yaw angles (excluding 0)", "yaw -60 to +60 (w/o 0)");

	// in der db: bei  ip.yaw=-60.0 fehlt 1 eintrag

	return 0;
}
