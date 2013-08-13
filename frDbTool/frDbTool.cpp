/*
 * frDbTool.cpp
 *
 *  Created on: 09.08.2013
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

#include "frDbTool.hpp"

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
            ("help,h", "Produce help message.")
			("create,c", "Create an empty database and generate the tables.") // dbfilename, 
			("add-imgdb,a", "Add an image-database to the 'database' table.") // dbfilename, name, info
			("import-images,i", "Import data from ... to the database.")
			("import-scores,s", "Import data from ... to the database.")
			("import-type-images,t", "The type of data to import. Currently implemented: -") // MPIE_FILELIST | ...
			("import-type-scores,r", "The type of data to import. Currently implemented: -") // MPIE_SCORES | ...
        ;

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).
                  options(desc).run(), vm);
        po::notify(vm);
    
        if (vm.count("help")) {
            cout << "[frDbTool] Usage: frDbTool [options]\n";
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

	bool doCreate = false;
	// create the database
	if (doCreate) {
		QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE");
		db.setDatabaseName(QString::fromStdString(databaseFilename));
		if (!db.open())	{
			cout << db.lastError().text().toStdString() << endl;
		}
		QSqlQuery query;
		bool ret;
		ret = query.exec("PRAGMA foreign_keys = ON");
		if(!ret) {
			cout << query.lastError().text().toStdString() << endl;
		}

		ret = query.exec("CREATE TABLE databases ( \
							name TEXT(25) NOT NULL, \
							info CLOB, \
							PRIMARY KEY (name) \
						)");
		if(!ret) {
			cout << query.lastError().text().toStdString() << endl;
		}

		ret = query.exec("CREATE TABLE images ( \
							 filepath CLOB NOT NULL, \
							 subject TEXT(25) NOT NULL, \
							 database TEXT(25) NOT NULL, \
							 session TEXT(25), \
							 roll REAL, \
							 pitch REAL, \
							 yaw REAL, \
							 lighting CLOB, \
							 expression CLOB, \
							 other CLOB, \
							 PRIMARY KEY (filepath), \
							 FOREIGN KEY (database) REFERENCES databases(name) \
						)");
		if(!ret) {
			cout << query.lastError().text().toStdString() << endl;
		}

		ret = query.exec("CREATE TABLE scores ( \
							probe TEXT(25) NOT NULL, \
							gallery TEXT(25) NOT NULL, \
							score REAL, \
							probeFailedToEnroll INTEGER, \
							algorithm TEXT(25), \
							FOREIGN KEY (probe) REFERENCES images(filepath), \
							FOREIGN KEY (gallery) REFERENCES images(filepath), \
							CONSTRAINT uniqueScores UNIQUE (probe, gallery, algorithm) \
						)");
		if(!ret) {
			cout << query.lastError().text().toStdString() << endl;
		}

		db.close();
		QSqlDatabase::removeDatabase("QSQLITE");
	}

	bool doAddImgDb = false;
	// add a new image-database to the db
	if (doAddImgDb) {
		QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE");
		db.setDatabaseName(QString::fromStdString(databaseFilename));
		if (!db.open())	{
			cout << db.lastError().text().toStdString() << endl;
		}
		QSqlQuery query;
		bool ret;
		ret = query.exec("PRAGMA foreign_keys = ON");
		if(!ret) {
			cout << query.lastError().text().toStdString() << endl;
		}

		ret = query.prepare("INSERT INTO databases (name, info) VALUES (:name, :info)");
		if(!ret) {
			cout << query.lastError().text().toStdString() << endl;
		}
		query.bindValue(":name", "multipie");
		query.bindValue(":info", "The MultiPIE image database.");
		ret = query.exec();
		if(!ret) {
			cout << query.lastError().text().toStdString() << endl;
		}
		db.close();
		QSqlDatabase::removeDatabase("QSQLITE");
	}

	bool doImportImages = false;
	if (doImportImages)	{
		// if type = multipie_filelist...

		QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE");
		db.setDatabaseName(QString::fromStdString(databaseFilename));
		if (!db.open())	{
			cout << db.lastError().text().toStdString() << endl;
		}
		QSqlQuery query;
		bool ret;
		ret = query.exec("PRAGMA foreign_keys = ON");
		if(!ret) {
			cout << query.lastError().text().toStdString() << endl;
		}

		//shared_ptr<ImageSource> imageSource = make_shared<FileListImageSource>("C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\mpie_frontal_gallery_fullpath.lst");
		shared_ptr<ImageSource> imageSource = make_shared<FileListImageSource>("C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\mpie_probe_fullpath.lst");
		while (imageSource->next()) {
			vector<size_t> slashes;
			size_t pos = imageSource->getName().string().find("/", 0);
			while(pos != string::npos)
			{
				slashes.push_back(pos);
				pos = imageSource->getName().string().find("/", pos+1);
			}
			size_t secondLastSlashPos = slashes[slashes.size()-2];
			size_t lastSlashPos = slashes[slashes.size()-1];
			string imagePath = imageSource->getName().string().substr(secondLastSlashPos+1, string::npos);
			string subject = imageSource->getName().stem().string().substr(0, 3);
			string database = "multipie";
			string session = "01"; // could be read from the filename?
			string lighting = "07"; // could be read from the filename?
			string expression = "neutral"; // could be read from the filename?
			string other = "";
			// extract the angles from the filename:
			string imageFn = imageSource->getName().string().substr(lastSlashPos+1, string::npos);
			
			vector<size_t> underlines;
			pos = imageFn.find("_", 0);
			while(pos != string::npos)
			{
				underlines.push_back(pos);
				pos = imageFn.find("_", pos+1);
			}
			string camera = imageFn.substr(underlines[underlines.size()-2]+1, 3);
			float roll = 0.0f;
			float pitch = 0.0f;
			float yaw;
			if (camera == "090")	{
				yaw = -60.0f;
			} else if (camera == "080") {
				yaw = -45.0f;
			} else if (camera == "130") { // subject looking right
				yaw = -30.0f;
			} else if (camera == "140") {
				yaw = -15.0f;
			} else if (camera == "051") { // frontal
				yaw = 0.0f;
			} else if (camera == "050") {
				yaw = 15.0f;
			} else if (camera == "041") { // subject looking left = positive = MPEG-standard
				yaw = 30.0f;
			} else if (camera == "190") {
				yaw = 45.0f;
			} else if (camera == "200") {
				yaw = 60.0f;
			}
			
			// insert the record into the database
			ret = query.prepare("INSERT INTO images (filepath, subject, database, session, roll, pitch, yaw, lighting, expression, other) VALUES (:filepath, :subject, :database, :session, :roll, :pitch, :yaw, :lighting, :expression, :other)");
			if(!ret) {
				cout << query.lastError().text().toStdString() << endl;
			}
			query.bindValue(":filepath", QVariant(QString::fromStdString(imagePath)));
			query.bindValue(":subject", QVariant(QString::fromStdString(subject)));
			query.bindValue(":database", QVariant(QString::fromStdString(database)));
			query.bindValue(":session", QVariant(QString::fromStdString(session)));
			query.bindValue(":roll", roll);
			query.bindValue(":pitch", pitch);
			query.bindValue(":yaw", yaw);
			query.bindValue(":lighting", QVariant(QString::fromStdString(lighting)));
			query.bindValue(":expression", QVariant(QString::fromStdString(expression)));
			query.bindValue(":other", QVariant(QString::fromStdString(other)));
			ret = query.exec();
			if(!ret) {
				cout << query.lastError().text().toStdString() << endl;
			}
			
		}
		db.close();
		QSqlDatabase::removeDatabase("QSQLITE");
	}

	bool doImportScores = true;
	if (doImportScores)	{
		// if type = ...

		QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE");
		db.setDatabaseName(QString::fromStdString(databaseFilename));
		if (!db.open())	{
			cout << db.lastError().text().toStdString() << endl;
		}
		QSqlQuery query;
		bool ret;
		ret = query.exec("PRAGMA foreign_keys = ON");
		if(!ret) {
			cout << query.lastError().text().toStdString() << endl;
		}

		path scoreOutDir = "C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\scores\\";
		// Import scores in the following format: A folder with the probe-image as filename, and the scores of all the galleries in the text file. We rely on the file-list to get the correct canonical name for the probe image.
		shared_ptr<ImageSource> imageSource = make_shared<FileListImageSource>("C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\mpie_probe_fullpath.lst");
		string algorithmName = "Full_FVSDK_8_7_0";
		while (imageSource->next()) {
			cout << imageSource->getName() << endl;
			vector<size_t> slashes;
			size_t pos = imageSource->getName().string().find("/", 0);
			while(pos != string::npos)
			{
				slashes.push_back(pos);
				pos = imageSource->getName().string().find("/", pos+1);
			}
			size_t secondLastSlashPos = slashes[slashes.size()-2];
			size_t lastSlashPos = slashes[slashes.size()-1];
			string imagePath = imageSource->getName().string().substr(secondLastSlashPos+1, string::npos);
			string imageCanonicalBasePath = imageSource->getName().string().substr(secondLastSlashPos+1, 25);
			string subject = imageSource->getName().stem().string().substr(0, 3);

			// Go through each probe image and calculate the statistics.
			string scoreOutPath = scoreOutDir.string() + imageSource->getName().stem().string() + ".txt";
			ifstream scoreFile(scoreOutPath);
			string line;
			if (scoreFile.is_open()) {
				path galleryName;
				float galleryScore;
				string galleryFteText;
				int galleryFteFlag;

				while (scoreFile.good())
				{
					getline(scoreFile, line);
					if (line == "")	{
						continue;
					}
					istringstream lineStream(line);
					lineStream >> galleryName;
					lineStream >> galleryScore;
					lineStream >> galleryFteText;

					ret = query.prepare("INSERT INTO scores (probe, gallery, score, probeFailedToEnroll, algorithm) VALUES (:probe, :gallery, :score, :probeFailedToEnroll, :algorithm)");
					if(!ret) {
						cout << query.lastError().text().toStdString() << endl;
					}
					string galleryImageCanonicalName = imageCanonicalBasePath + galleryName.stem().string() + ".png";
					if (galleryFteText == "FTE") {
						galleryFteFlag = 1;
					} else {
						galleryFteFlag = 0;
					}
					query.bindValue(":probe", QVariant(QString::fromStdString(imagePath)));
					query.bindValue(":gallery", QVariant(QString::fromStdString(galleryImageCanonicalName)));
					query.bindValue(":score", galleryScore);
					query.bindValue(":probeFailedToEnroll", galleryFteFlag);
					query.bindValue(":algorithm", QVariant(QString::fromStdString(algorithmName)));
					ret = query.exec();
					if(!ret) {
						cout << query.lastError().text().toStdString() << endl;
					}
				}
				scoreFile.close();

			}

		}
		db.close();
		QSqlDatabase::removeDatabase("QSQLITE");
	}


	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(loglevel::TRACE));

	path fvsdkBins = "C:\\Users\\Patrik\\Cloud\\PhD\\FVSDK_bins\\";
	path firOutDir = "C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\FIRs\\";
	path scoreOutDir = "C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\scores\\";
	path galleryFirList = "C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\mpie_frontal_gallery_fullpath_firlist.lst";

	
	
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

	// 

	// Go through each probe image and calculate the statistics.
	/*
	int numProbes = 0;
	int fte = 0;
	int rank1Matches = 0;
	int noRank1Match = 0;
	while (probeImageSource->next()) {
		string scoreOutPath = scoreOutDir.string() + probeImageSource->getName().stem().string() + ".txt";
		ifstream scoreFile(scoreOutPath);
		string line;
		if (scoreFile.is_open()) {
			getline(scoreFile, line);
			++numProbes;
			if (line == "FTE") {	// is it a probe that couldn't be enrolled?
				++fte;
				scoreFile.close();
				continue;
			}
			// we were able to enroll the probe, go on:
			//auto comp = [](const pair<string, float>& lhs, const pair<string, float>& rhs) -> bool { return lhs.second < rhs.second; };
			//auto galleryScores = std::set<pair<string, float>, decltype(comp)>(comp);
			typedef set< pair<string, float>, function<bool(pair<string, float>, pair<string, float>)> > ScoresSet;
			ScoresSet galleryScores(
				[] ( pair<string, float> lhs, pair<string, float> rhs ) { return lhs.second > rhs.second ; } ) ;

			path galleryName;
			float galleryScore;
			istringstream lineStream(line);
			lineStream >> galleryName;
			lineStream >> galleryScore;
			galleryScores.insert(make_pair(galleryName.stem().string(), galleryScore));
			while (scoreFile.good())
			{
				getline(scoreFile, line);
				istringstream lineStream(line);
				lineStream >> galleryName;
				lineStream >> galleryScore;
				galleryScores.insert(make_pair(galleryName.stem().string(), galleryScore));
			}
			scoreFile.close();

			// Go see if we got a match (rank-1):
			cout << "Gallery rank-1 match: " << galleryScores.begin()->first << ", Probe: " << probeImageSource->getName().stem().string() << endl;
			if (subjectsMatch(galleryScores.begin()->first, probeImageSource->getName().stem().string())) {
				++rank1Matches;
			} else {
				++noRank1Match;
			}
		}
		
		
	}

	cout << numProbes << ", " << fte << ", " << rank1Matches << ", " << noRank1Match << endl;
	*/

	// map flip
	// unordered map, sortiert einfuegen (letztes element merken)
	// min/max_element
	// std::set(=ordered) mit pair und custom comparator
	// boost::bimap

	return 0;
}
