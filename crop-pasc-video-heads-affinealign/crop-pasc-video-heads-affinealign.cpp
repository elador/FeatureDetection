/*
 * crop-pasc-video-heads-affinealign.cpp
 *
 *  Created on: 28.10.2014
 *      Author: Patrik Huber
 *
 * Example:
 * crop-pasc-video-heads-affinealign ...
 *   
 */

#include <memory>
#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/range.hpp"
#include "boost/range/algorithm.hpp"
#include "boost/range/adaptor/filtered.hpp"

#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"
#include "facerecognition/alignment.hpp"

#include "imageio/SimpleModelLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace facerecognition;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using boost::filesystem::path;
using cv::Mat;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::make_shared;

using namespace cv;

// Draws the visible (i.e. inside the image boundaries) triangles.
void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		if (t[0] >= img.cols || t[1] >= img.rows || t[2] >= img.cols || t[3] >= img.rows || t[4] >= img.cols || t[5] >= img.rows) {
			continue;
		}
		if (t[0] < 0 || t[1] < 0 || t[2] < 0 || t[3] < 0 || t[4] < 0 || t[5] < 0) {
			continue;
		}
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
		line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
		line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
	}
}

// Paints the Voronoi diagram of the found triangles
void paint_voronoi(Mat& img, Subdiv2D& subdiv)
{
	vector<vector<Point2f> > facets;
	vector<Point2f> centers;
	subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

	vector<Point> ifacet;
	vector<vector<Point> > ifacets(1);

	for (size_t i = 0; i < facets.size(); i++)
	{
		ifacet.resize(facets[i].size());
		for (size_t j = 0; j < facets[i].size(); j++)
			ifacet[j] = facets[i][j];

		Scalar color;
		color[0] = rand() & 255;
		color[1] = rand() & 255;
		color[2] = rand() & 255;
		fillConvexPoly(img, ifacet, color, 8, 0);

		ifacets[0] = ifacet;
		polylines(img, ifacets, true, Scalar(), 1, CV_AA, 0);
		circle(img, centers[i], 3, Scalar(), -1, CV_AA, 0);
	}
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path sigsetFile, metadataFile, landmarksDirectory, inputDirectory;
	path outputFolder;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
//			("sigset,s", po::value<path>(&sigsetFile)->required(),
//				"PaSC video XML sigset of the frames to be cropped")
			("input,i", po::value<path>(&inputDirectory)->required(),
				"directory containing the frames. Files should be in the format 'videoname.012.png'")
			("metadata,m", po::value<path>(&metadataFile)->required(),
				"PaSC video detections metadata in boost::serialization format")
			("landmarks,l", po::value<path>(&landmarksDirectory)->required(),
				"Directory containing landmarks to perform the piecewise affine alignment with")
			("output,o", po::value<path>(&outputFolder)->default_value("."),
				"path to save the cropped patches to")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: crop-pasc-video-heads-affinealign [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);

	}
	catch (po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	LogLevel logLevel;
	if(boost::iequals(verboseLevelConsole, "PANIC")) logLevel = LogLevel::Panic;
	else if(boost::iequals(verboseLevelConsole, "ERROR")) logLevel = LogLevel::Error;
	else if(boost::iequals(verboseLevelConsole, "WARN")) logLevel = LogLevel::Warn;
	else if(boost::iequals(verboseLevelConsole, "INFO")) logLevel = LogLevel::Info;
	else if(boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = LogLevel::Debug;
	else if(boost::iequals(verboseLevelConsole, "TRACE")) logLevel = LogLevel::Trace;
	else {
		cout << "Error: Invalid LogLevel." << endl;
		return EXIT_FAILURE;
	}
	
	Loggers->getLogger("facerecognition").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	if (!fs::exists(inputDirectory) || fs::is_regular_file(inputDirectory)) {
		return EXIT_FAILURE;
	}
	
	// Create the output directory if it doesn't exist yet
	if (!boost::filesystem::exists(outputFolder)) {
		boost::filesystem::create_directory(outputFolder);
	}

	// Read the video detections metadata (eyes, face-coords):
	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(metadataFile.string());
		boost::archive::text_iarchive ia(ifs);
		ia >> pascVideoDetections;
	}

	//auto sigset = facerecognition::utils::readPascSigset(sigsetFile, true);
	vector<path> files;
	try	{
		fs::directory_iterator end_itr;
		for (boost::filesystem::directory_iterator i(inputDirectory); i != end_itr; ++i)
		{
			if (fs::is_regular_file(i->status()) && i->path().extension() == ".png")
				files.emplace_back(i->path());
		}
	}
	catch (const fs::filesystem_error& ex)
	{
		cout << ex.what() << endl;
	}

	FivePointModel model;

	for (auto& file : files) {
		appLogger.info("Processing " + file.string());

		string frameNumExtension = file.stem().extension().string(); // from videoname.123.png to ".123"
		frameNumExtension.erase(std::remove(frameNumExtension.begin(), frameNumExtension.end(), '.'), frameNumExtension.end());
		int frameNum = boost::lexical_cast<int>(frameNumExtension);
		path videoName = file.stem().stem(); // from videoname.123.png to "videoname"
		videoName.replace_extension(".mp4");
		string frameName = facerecognition::getPascFrameName(videoName, frameNum);
		
		// Load our own landmark detections:
		auto landmarksFile = landmarksDirectory / file.stem().stem() / file.filename();
		landmarksFile.replace_extension(".txt");
		imageio::SimpleModelLandmarkFormatParser lmParser;
		// Todo: This can crash, if we don't have PP eyes, we won't have our own LMs!
		auto lms = lmParser.read(landmarksFile).at(file.stem().string());

		Mat frame = cv::imread(file.string());

		// Now the same for the detected landmarks:
		vector<Point2f> landmarkPoints;
		for (auto& l : lms.getLandmarks()) {
			landmarkPoints.emplace_back(l->getPoint2D()); // order: re_c, le_c, mouth_c, nt, botn
		}
		landmarkPoints = addArtificialPoints(landmarkPoints);

		// Now warp the detected landmarks triangles to the reference (mean), using a triangle-wise (piecewise) affine mapping:
		Mat textureMap = model.extractTexture2D(frame, landmarkPoints);
			
		//string logMessage("Throwing away patch because it goes outside the image bounds. This shouldn't happen, or rather, we do not want it to happen, because we didn't select a frame where this should happen.");
		//string logMessage("Hmm no PP eyes. How are we going to do the affine alignment? Skipping the image for now.");

		// Normalise:
		//textureMap = utils::equaliseIntensity(textureMap);
		
		/*
		Mat quotientImage;
		cv::cvtColor(textureMap, textureMap, CV_RGB2GRAY);
		textureMap.convertTo(textureMap, CV_32FC1, 1.0 / 255);
		cv::GaussianBlur(textureMap, quotientImage, cv::Size(7, 7), 1.0);
		quotientImage = textureMap / quotientImage;
		auto mean = cv::mean(quotientImage);
		quotientImage = quotientImage - mean[0];
		*/

		path croppedImageFilename = outputFolder / file.filename();
		cv::imwrite(croppedImageFilename.string(), textureMap);
	}
	appLogger.info("Finished cropping all files in the directory.");

	return EXIT_SUCCESS;
}
