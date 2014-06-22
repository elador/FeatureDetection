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

#include "facerecognition/FaceRecord.hpp"
#include "facerecognition/utils.hpp"

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

	string databaseFilename = "C:\\Users\\Patrik\\Documents\\Github\\experiments\\frdb.sqlite";

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
		ret = query.exec("CREATE TABLE records ( \
							 identifier TEXT(25) NOT NULL, \
							 datapath CLOB NOT NULL, \
							 subject TEXT(25) NOT NULL, \
							 database TEXT(25) NOT NULL, \
							 session TEXT(25), \
							 roll REAL, \
							 pitch REAL, \
							 yaw REAL, \
							 lighting CLOB, \
							 expression CLOB, \
							 other CLOB, \
							 PRIMARY KEY (identifier), \
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
							FOREIGN KEY (probe) REFERENCES records(identifier), \
							FOREIGN KEY (gallery) REFERENCES records(identifier), \
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
		query.bindValue(":name", "MultiPIE");
		query.bindValue(":info", "The MultiPIE image database.");
		//query.bindValue(":name", "PaSC");
		//query.bindValue(":info", "The Point and Shoot Face Recognition Challenge image and video database.");
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

		path sigsetToImport(R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\lists\gallery_frontal.sig.txt)");
		auto sigset = facerecognition::utils::readSigset(sigsetToImport);
		for (auto&& record : sigset) {
			string database = "MultiPIE";
			// insert the record into the database:
			ret = query.prepare("INSERT INTO records (identifier, datapath, subject, database, session, roll, pitch, yaw, lighting, expression, other) VALUES (:identifier, :datapath, :subject, :database, :session, :roll, :pitch, :yaw, :lighting, :expression, :other)");
			if(!ret) {
				cout << query.lastError().text().toStdString() << endl;
			}
			query.bindValue(":identifier", QVariant(QString::fromStdString(record.identifier)));
			query.bindValue(":datapath", QVariant(QString::fromStdString(record.dataPath.string())));
			query.bindValue(":subject", QVariant(QString::fromStdString(record.subjectId)));
			query.bindValue(":database", QVariant(QString::fromStdString(database)));
			query.bindValue(":session", QVariant(QString::fromStdString(record.session)));
			if (record.roll) {
				query.bindValue(":roll", record.roll.get());
			}
			else {
				query.bindValue(":roll", "");
			}
			if (record.pitch) {
				query.bindValue(":pitch", record.pitch.get());
			}
			else {
				query.bindValue(":pitch", "");
			}
			if (record.yaw) {
				query.bindValue(":yaw", record.yaw.get());
			}
			else {
				query.bindValue(":yaw", "");
			}
			query.bindValue(":lighting", QVariant(QString::fromStdString(record.lighting)));
			query.bindValue(":expression", QVariant(QString::fromStdString(record.expression)));
			query.bindValue(":other", QVariant(QString::fromStdString(record.other)));
			ret = query.exec();
			if(!ret) {
				cout << query.lastError().text().toStdString() << endl;
			}
			
		}
		db.close();
		QSqlDatabase::removeDatabase("QSQLITE");
	}

	// Import the old MultiPIE scores of the first baseline experiment:
	bool doImportOldMpieScores = true;
	if (doImportOldMpieScores)	{
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
			//string imagePath = imageSource->getName().string().substr(secondLastSlashPos+1, string::npos);
			path imagePath = imageSource->getName();
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
					if (galleryFteText == "FTE") {
						galleryFteFlag = 1;
					} else {
						galleryFteFlag = 0;
					}
					query.bindValue(":probe", QVariant(QString::fromStdString(imagePath.stem().string())));
					query.bindValue(":gallery", QVariant(QString::fromStdString(galleryName.stem().string())));
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
	
	return 0;
}
