/*
 * frRunCogExp.cpp
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

#include "frRunCogExp.hpp"

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

#include <frsdk/config.h>
#include <frsdk/match.h>

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <QtSql>

#include <iostream>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <iterator>

namespace po = boost::program_options;
using namespace std;
using namespace imageio;
using namespace facerecognition;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
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

	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(LogLevel::Trace));
	Loggers->getLogger("facerecognition").addAppender(make_shared<logging::ConsoleAppender>(LogLevel::Trace));

	path fvsdkBins = R"(C:\Users\Patrik\Documents\GitHub\FVSDK_source\Release\)";
	
	//path scoreOutDir = R"(C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\scores\\)";
	//path galleryFirList = R"(C:\\Users\\Patrik\\Documents\\GitHub\\data\\mpie_pics_paper_FRexp\\mpie_frontal_gallery_fullpath_firlist.lst)";

	// The exact images from the list are used. Output FIR is firOutDir plus the basename of the image.
	//shared_ptr<FileListImageSource> imageList = make_shared<FileListImageSource>(R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\lists\probe_p15.txt)");
	//path firOutDir = R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\02_27062014_TexExtr_visCheck\FIRs\probe_p15\)";

	//Mat img;
	//string cmd;
	// create all FIRs of images
	/*
	while (imageList->next()) {
		//img = imageList->getImage();
		cmd = fvsdkBins.string() + "enroll.exe " + "-cfg C:\\FVSDK_8_9_5\\etc\\frsdk.cfg " + "-fir " + firOutDir.string() + imageList->getName().stem().string() + ".fir " + "-imgs " + imageList->getName().string();
		int cmdRet = system(cmd.c_str());
		cout << cmdRet << endl;
	}*/

	// create all FIRs of probes, use imageList for the basename, but then use fittingsDir and append _isomap:
	/*
	path fittingsDir = R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\02_27062014_TexExtr_visCheck\fittings\probe_p15\)";
	while (imageList->next()) {
		path isomapImageFile = fittingsDir.string() + imageList->getName().stem().string() + "_render_frontal.png";
		cmd = fvsdkBins.string() + "enroll.exe " + "-cfg C:\\FVSDK_8_9_5\\etc\\frsdk.cfg " + "-fir " + firOutDir.string() + imageList->getName().stem().string() + ".fir " + "-imgs " + isomapImageFile.string();
		int cmdRet = system(cmd.c_str());
		cout << cmdRet << endl;
	}*/

	// Matching: Create all the scores of every probe against every gallery entry
	string algorithmName = "Fitting_CamShp_FrontalRendering_FVSDK_8_9_5";

	path transformConfigFile(R"(C:\Users\Patrik\Documents\GitHub\FeatureDetection\libFaceRecognition\share\config\transformRecordDataPath.txt)");
	ptree transformConfig;
	try {
		boost::property_tree::read_info(transformConfigFile.string(), transformConfig);
	}
	catch (boost::property_tree::info_parser_error& error) {
		string errorMessage{ string("Error reading the transformation config file: ") + error.what() };
		//appLogger.error(errorMessage);
		throw error;
	}
	utils::DataPathTransformation transformProbe = utils::DataPathTransformation::read(transformConfig.get_child("probe"));
	utils::DataPathTransformation transformGallery = utils::DataPathTransformation::read(transformConfig.get_child("gallery"));

	auto probe = utils::readSigset(R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\lists\probe_p60.sig.txt)");
	auto gallery = utils::readSigset(R"(C:\Users\Patrik\Documents\GitHub\experiments\MultiPIE\lists\gallery_frontal.sig.txt)");

	// Open the database
	path databaseFilename(R"(C:\Users\Patrik\Documents\Github\experiments\frdb.sqlite)");
	QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE");
	db.setDatabaseName(QString::fromStdString(databaseFilename.string()));
	if (!db.open())	{
		cout << db.lastError().text().toStdString() << endl;
	}
	QSqlQuery query;
	bool ret;
	ret = query.exec("PRAGMA foreign_keys = ON");
	if (!ret) {
		cout << query.lastError().text().toStdString() << endl;
	}

	// Start the FVSDK-stuff
	try {
		// We first enroll the whole gallery:
		path frsdkConfig(R"(C:\FVSDK_8_9_5\etc\frsdk.cfg)");
		// initialize and resource allocation
		FRsdk::Configuration cfg(frsdkConfig.string());
		FRsdk::FIRBuilder firBuilder(cfg);
		// initialize matching facility
		FRsdk::FacialMatchingEngine me(cfg);
		// load the fir population (gallery)
		FRsdk::Population population(cfg);
		for (auto&& g : gallery) {
			path galleryFilename = g.dataPath;
			galleryFilename = utils::transformDataPath(galleryFilename, transformGallery);
			ifstream firIn(galleryFilename.string(), ios::in | ios::binary);
			population.append(firBuilder.build(firIn), galleryFilename.string().c_str());
		}

		// do the matching:
		for (auto&& p : probe) {
			cout << "Matching " << p.identifier << endl;
			path probeFilename = p.dataPath;
			probeFilename = utils::transformDataPath(probeFilename, transformProbe);
			// We have our probe and gallery FIR paths, go match them:
			// load the fir of the probe
			if (boost::filesystem::exists(probeFilename)) { // The FIR exists, meaning we were able to enroll a probe, go and match it against gallery
				ifstream firStream(probeFilename.string(), ios::in | ios::binary);
				FRsdk::FIR fir = firBuilder.build(firStream);

				// Compare the probe against the gallery and get a score for each gallery entry:
				FRsdk::CountedPtr<FRsdk::Scores> scores = me.compare(fir, population);
				int idx = 0;
				for (FRsdk::Scores::const_iterator siter = scores->begin(); siter != scores->end(); siter++) {
					auto score = static_cast<float>(*siter);
					ret = query.prepare("INSERT INTO scores (probe, gallery, score, probeFailedToEnroll, algorithm) VALUES (:probe, :gallery, :score, :probeFailedToEnroll, :algorithm)");
					if (!ret) {
						cout << query.lastError().text().toStdString() << endl;
					}
					query.bindValue(":probe", QVariant(QString::fromStdString(p.identifier)));
					query.bindValue(":gallery", QVariant(QString::fromStdString(gallery[idx].identifier)));
					query.bindValue(":score", score);
					query.bindValue(":probeFailedToEnroll", 0);
					query.bindValue(":algorithm", QVariant(QString::fromStdString(algorithmName)));
					ret = query.exec();
					if (!ret) {
						cout << query.lastError().text().toStdString() << endl;
					}
					++idx;
				}
			}
			else { // We were not able to enroll the probe
				for (auto&& g : gallery) {
					ret = query.prepare("INSERT INTO scores (probe, gallery, score, probeFailedToEnroll, algorithm) VALUES (:probe, :gallery, :score, :probeFailedToEnroll, :algorithm)");
					if (!ret) {
						cout << query.lastError().text().toStdString() << endl;
					}
					query.bindValue(":probe", QVariant(QString::fromStdString(p.identifier)));
					query.bindValue(":gallery", QVariant(QString::fromStdString(g.identifier)));
					query.bindValue(":score", 0.0f);
					query.bindValue(":probeFailedToEnroll", 1);
					query.bindValue(":algorithm", QVariant(QString::fromStdString(algorithmName)));
					ret = query.exec();
					if (!ret) {
						cout << query.lastError().text().toStdString() << endl;
					}
				}
			}

		}

	}
	catch (const FRsdk::FeatureDisabled& e) {
		cout << "Feature not enabled: " << e.what() << endl;
		db.close();
		QSqlDatabase::removeDatabase("QSQLITE");
		return EXIT_FAILURE;
	}
	catch (const FRsdk::LicenseSignatureMismatch& e) {
		cout << "License violation: " << e.what() << endl;
		db.close();
		QSqlDatabase::removeDatabase("QSQLITE");
		return EXIT_FAILURE;
	}
	catch (exception& e) {
		cout << e.what() << endl;
		db.close();
		QSqlDatabase::removeDatabase("QSQLITE");
		return EXIT_FAILURE;
	}
	db.close();
	QSqlDatabase::removeDatabase("QSQLITE");
	return 0;
}
