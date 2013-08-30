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

void toMatlab(path filename, vector<float> scores, int numFailureToEnroll, string plotTitle, string curveName, string curveParameters="r-", bool appendToFile=false) {

	ofstream file;

	if (appendToFile)
		file.open(filename.string(), ios::app);
	else
		file.open(filename.string());

	if (file.is_open()) {

		string matlabVarName = curveName;
		replace(begin(matlabVarName), end(matlabVarName), ' ', '_');
		replace(begin(matlabVarName), end(matlabVarName), '-', '_');
		replace(begin(matlabVarName), end(matlabVarName), '+', '_');
		replace(begin(matlabVarName), end(matlabVarName), '=', '_');
		replace(begin(matlabVarName), end(matlabVarName), '(', '_');
		replace(begin(matlabVarName), end(matlabVarName), ')', '_');
		replace(begin(matlabVarName), end(matlabVarName), '.', '_');
		replace(begin(matlabVarName), end(matlabVarName), ',', '_');

		if (appendToFile)
			file << "hold on;" << endl;
		if (numFailureToEnroll > 0) {
			file << matlabVarName << "_y = 0:(1/(size(" << matlabVarName << ", 2)+numFte - 1)):1;" << endl;
			file << "numFte = " << numFailureToEnroll << ";" << endl; // Todo: Check: If I call this function twice, with different FTE-vals (can that happen?), do I get the right curves?
			// Todo: Check for FTE = 0?
			file << "numFteVec = zeros(1, " << numFailureToEnroll << ");" << endl; // or, if the score could get negative, chose the point a little bit behind the lowest point's score
			file << "fte_yAxisVal = numFte/size([numFteVec, " << matlabVarName << "], 2);" << endl; // the y-value (the FTE-percentage), our curve crosses the y-axis here and starts here
			file << "fte_yAxisValVec = fte_yAxisVal * ones(1, numFte);" << endl;
			file << matlabVarName << "_y(1:numFte) = fte_yAxisValVec;" << endl; // Replace the y-axis values by the fte-y-axis-val
			file << matlabVarName << " = [";
			copy(begin(scores), end(scores), ostream_iterator<float>(file, " "));
			file << "];" << endl;
			file << "plot([numFteVec, " << matlabVarName << "], " << matlabVarName << "_y, '" << curveParameters << "', 'DisplayName', '" << curveName << "');" << endl;
		} else {
			file << matlabVarName << "_y = 0:(1/(size(" << matlabVarName << ", 2) - 1)):1;" << endl;
			file << matlabVarName << " = [";
			copy(begin(scores), end(scores), ostream_iterator<float>(file, " "));
			file << "];" << endl;
			file << "plot(" << matlabVarName << ", " << matlabVarName << "_y, '" << curveParameters << "', 'DisplayName', '" << curveName << "');" << endl;
		}
		file << "legend('-DynamicLegend');" << endl;
		file << "title('" << plotTitle << "');" << endl;
		file << "xlabel('Score');" << endl;
		file << "ylabel('Frequency');" << endl; // Percentage of examples

		file.close();

	}
}

void toMatlabDetCurve(path filename, string plotTitle, string curveNameFp, string curveNameFn, string curveParameters="r-") {

	ofstream file;
	file.open(filename.string(), ios::app);
	
	if (file.is_open()) {

		string matlabVarNameFp = curveNameFp;
		replace(begin(matlabVarNameFp), end(matlabVarNameFp), ' ', '_');
		replace(begin(matlabVarNameFp), end(matlabVarNameFp), '-', '_');
		replace(begin(matlabVarNameFp), end(matlabVarNameFp), '+', '_');
		replace(begin(matlabVarNameFp), end(matlabVarNameFp), '=', '_');
		replace(begin(matlabVarNameFp), end(matlabVarNameFp), '(', '_');
		replace(begin(matlabVarNameFp), end(matlabVarNameFp), ')', '_');
		replace(begin(matlabVarNameFp), end(matlabVarNameFp), '.', '_');
		replace(begin(matlabVarNameFp), end(matlabVarNameFp), ',', '_');

		string matlabVarNameFn = curveNameFn;
		replace(begin(matlabVarNameFn), end(matlabVarNameFn), ' ', '_');
		replace(begin(matlabVarNameFn), end(matlabVarNameFn), '-', '_');
		replace(begin(matlabVarNameFn), end(matlabVarNameFn), '+', '_');
		replace(begin(matlabVarNameFn), end(matlabVarNameFn), '=', '_');
		replace(begin(matlabVarNameFn), end(matlabVarNameFn), '(', '_');
		replace(begin(matlabVarNameFn), end(matlabVarNameFn), ')', '_');
		replace(begin(matlabVarNameFn), end(matlabVarNameFn), '.', '_');
		replace(begin(matlabVarNameFn), end(matlabVarNameFn), ',', '_');

		file << "figure(2);" << endl;
		file << "[det_fp, det_fn] = fpfn2detCurve(" << matlabVarNameFp << ", " << matlabVarNameFp << "_y, " << matlabVarNameFn << ", " << matlabVarNameFn << "_y);" << endl;
		file << "plot(det_fp, det_fn, '" << curveParameters << "', 'DisplayName', '" << "DetCurve" << "');" << endl;
		file << "legend('-DynamicLegend');" << endl;
		file << "title('" << plotTitle << "');" << endl;
		file << "xlabel('FP');" << endl;
		file << "ylabel('FN');" << endl;

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
	ret = query.exec("SELECT s.* FROM scores s INNER JOIN images ip ON s.probe = ip.filepath INNER JOIN images ig ON s.gallery = ig.filepath WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=45.0 OR ip.yaw=-45.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject=ig.subject"); // (if I had more images, add WHERE db=multipie, lighting=..., etc.) // AND (ip.yaw=60.0 OR ip.yaw=45) // AND ip.subject=001
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

	ret = query.exec("SELECT s.* FROM scores s INNER JOIN images ip ON s.probe = ip.filepath INNER JOIN images ig ON s.gallery = ig.filepath WHERE s.algorithm='Full_FVSDK_8_7_0' AND ip.roll=0.0 AND ip.pitch=0.0 AND (ip.yaw=45.0 OR ip.yaw=-45.0) AND ig.roll=0.0 AND ig.pitch=0.0 AND ig.yaw=0.0 AND ip.subject!=ig.subject"); // (if I had more images, add WHERE db=multipie, lighting=..., etc.) // AND (ip.yaw=60.0 OR ip.yaw=45) // AND ip.subject=001
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

	toMatlab("C:/Users/Patrik/Documents/MATLAB/scores.m", positiveScores, numFailureToEnroll, "Cumulative histogram of the scores of the positive and negative ground-truth",  "FRR (positives) y=45", "r-", false);
	toMatlab("C:/Users/Patrik/Documents/MATLAB/scores.m", negativeScores, 0, "Cumulative histogram of the scores of the positive and negative ground-truth",  "FAR (negatives) y=45", "b-", true);

	toMatlabDetCurve("C:/Users/Patrik/Documents/MATLAB/scores.m", "Det", "FRR (positives) y=45", "FAR (negatives) y=45", "r-");

	// toMatlab(void) (generate a matlab string)
	// toAscii(string fn) (generate a ascii (csv or similar) to plot in ML)
	// own plot fuction

	// in der db: bei  ip.yaw=-60.0 fehlt 1 eintrag

	return 0;
}
