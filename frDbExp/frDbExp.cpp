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
using logging::LogLevel;
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

	string databaseFilename = "C:\\Users\\Patrik\\Documents\\Github\\experiments\\frdb.sqlite";

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
	// Probes all +-15, 30, 45 and 60:
	//ret = query.exec("SELECT s.* FROM scores s INNER JOIN records ip ON s.probe = ip.identifier INNER JOIN records ig ON s.gallery = ig.identifier WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0 OR ip.yaw=-15.0 OR ip.yaw=30.0 OR ip.yaw=-30.0 OR ip.yaw=45.0 OR ip.yaw=-45.0 OR ip.yaw=60.0 OR ip.yaw=-60.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject=ig.subject"); // (if I had more images, add WHERE db=multipie, lighting=..., etc.) // AND (ip.yaw=60.0 OR ip.yaw=45) // AND ip.subject=001
	// Probes all +15, 30, 45 and 60:
	ret = query.exec("SELECT s.* FROM scores s INNER JOIN records ip ON s.probe = ip.identifier INNER JOIN records ig ON s.gallery = ig.identifier WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0 OR ip.yaw=30.0 OR ip.yaw=45.0 OR ip.yaw=60.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject=ig.subject");
	// Probes +xx:
	//ret = query.exec("SELECT s.* FROM scores s INNER JOIN records ip ON s.probe = ip.identifier INNER JOIN records ig ON s.gallery = ig.identifier WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject=ig.subject");
	// algorithm: 'Full_FVSDK_8_7_0' or 'Fitting_CamShp_FrontalRendering_FVSDK_8_9_5'

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

	// Probes all +-15, 30, 45 and 60:
	//ret = query.exec("SELECT s.* FROM scores s INNER JOIN records ip ON s.probe = ip.identifier INNER JOIN records ig ON s.gallery = ig.identifier WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0 OR ip.yaw=-15.0 OR ip.yaw=30.0 OR ip.yaw=-30.0 OR ip.yaw=45.0 OR ip.yaw=-45.0 OR ip.yaw=60.0 OR ip.yaw=-60.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject!=ig.subject"); // (if I had more images, add WHERE db=multipie, lighting=..., etc.) // AND (ip.yaw=60.0 OR ip.yaw=45) // AND ip.subject=001
	// Probes all +15, 30, 45 and 60:
	ret = query.exec("SELECT s.* FROM scores s INNER JOIN records ip ON s.probe = ip.identifier INNER JOIN records ig ON s.gallery = ig.identifier WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0 OR ip.yaw=30.0 OR ip.yaw=45.0 OR ip.yaw=60.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject!=ig.subject");
	// Probes +xx:
	//ret = query.exec("SELECT s.* FROM scores s INNER JOIN records ip ON s.probe = ip.identifier INNER JOIN records ig ON s.gallery = ig.identifier WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject!=ig.subject");
	// ip = image probe, ig = image gallery
	if (!ret) {
		cout << query.lastError().text().toStdString() << endl;
	}
	while (query.next()) {
		if (query.value(3).toInt() != 1) {
			negativeScores.push_back(query.value(2).toFloat());
		}
	}
	
	
	// For the DET figure and verification rates
	sort(begin(positiveScores), end(positiveScores), less<float>()); // 0 to 1
	sort(begin(negativeScores), end(negativeScores), greater<float>()); // 1 to 0

	//writeAscii("C:/Users/Patrik/Documents/Github/experiments/MultiPIE/00_00002013_FaceVACS_8_7_0_Baseline/scores/pm15_30_45_60.txt", positiveScores, negativeScores, numFailureToEnroll, "Probes +-15, +-30, +-45, +-60 yaw, with FTEs", "yaw=+-15,30,45,60");
	writeAscii("C:/Users/Patrik/Documents/Github/experiments/MultiPIE/00_00002013_FaceVACS_8_7_0_Baseline/scores/p15_30_45_60.txt", positiveScores, negativeScores, numFailureToEnroll, "Probes +15, +30, +45, +60 yaw, with FTEs", "yaw=+15,30,45,60");
	//writeAscii("C:/Users/Patrik/Documents/Github/experiments/MultiPIE/00_00002013_FaceVACS_8_7_0_Baseline/scores/p15.txt", positiveScores, negativeScores, numFailureToEnroll, "Probes +15, with FTEs", "yaw=+15");

	// Calculate the rank-1 identification rate
	//ret = query.exec("SELECT s.*, MAX(s.score) FROM scores s INNER JOIN records ip ON s.probe = ip.identifier INNER JOIN records ig ON s.gallery = ig.identifier WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0 OR ip.yaw=-15.0 OR ip.yaw=30.0 OR ip.yaw=-30.0 OR ip.yaw=45.0 OR ip.yaw=-45.0 OR ip.yaw=60.0 OR ip.yaw=-60.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 GROUP BY s.probe");
	ret = query.exec("SELECT s.*, MAX(s.score) FROM scores s INNER JOIN records ip ON s.probe = ip.identifier INNER JOIN records ig ON s.gallery = ig.identifier WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0 OR ip.yaw=30.0 OR ip.yaw=45.0 OR ip.yaw=60.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 GROUP BY s.probe");
	//ret = query.exec("SELECT s.*, MAX(s.score) FROM scores s INNER JOIN records ip ON s.probe = ip.identifier INNER JOIN records ig ON s.gallery = ig.identifier WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=15.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 GROUP BY s.probe");
	if (!ret) {
		cout << query.lastError().text().toStdString() << endl;
	}
	int numProbes = 0;
	int fte = 0;
	int rank1Matches = 0;
	int noRank1Match = 0;
	while (query.next()) {
		++numProbes;
		path probe = query.value(0).toString().toStdString();
		path gallery = query.value(1).toString().toStdString();
		if (query.value(3).toInt() == 1) {
			++fte;
			continue;
		}
		if (subjectsMatch(probe.stem().string(), gallery.stem().string())) {
			++rank1Matches;
		} else {
			++noRank1Match;
		}
	}

	cout << "Probes: " << numProbes << ", FTE: " << fte << ", rank-1 matches: " << rank1Matches << ", no rank-1 matches: " << noRank1Match << endl;
	cout << "rank-1 id percent with fte: " << (float)rank1Matches / (float)numProbes << endl;
	cout << "rank-1 id percent without fte: " << (float)rank1Matches / ((float)numProbes - (float)fte) << endl;

	// For rank-n: SELECT TOP 5, or SELECT x from Y LIMIT 5, ... Not so easy. Better do it in the C++ code.
	// http://www.xaprb.com/blog/2006/12/07/how-to-select-the-firstleastmax-row-per-group-in-sql/
	// http://stackoverflow.com/questions/176964/select-top-10-records-for-each-category
	// http://stackoverflow.com/questions/2129693/mysql-using-limit-within-group-by-to-get-n-results-per-group
	// http://stackoverflow.com/questions/13091457/order-by-and-limit-in-group-by
	// http://pratchev.blogspot.co.uk/2008/11/top-n-by-group.html
	// http://stackoverflow.com/questions/16047394/trouble-with-parenthesis-in-ms-access-for-sql-inner-joins
	// http://stackoverflow.com/questions/4886448/mysql-group-by-max-value
	// http://stackoverflow.com/questions/1591909/group-by-behavior-when-no-aggregate-functions-are-present-in-the-select-clause
	// http://www.artfulsoftware.com/infotree/qrytip.php?id=104

	db.close();
	QSqlDatabase::removeDatabase("QSQLITE");

	// in der db: bei  ip.yaw=-60.0 fehlt 1 eintrag
	return 0;
}
