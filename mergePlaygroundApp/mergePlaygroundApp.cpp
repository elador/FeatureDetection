/*
 * mergePlaygroundApp.cpp
 *
 *  Created on: 15.06.2013
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
#include <memory>
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
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"

#include "classification/RbfKernel.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/WvmClassifier.hpp"
#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/ProbabilisticRvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/ReshapingFilter.hpp"
#include "imageprocessing/ConversionFilter.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/FilteringPyramidFeatureExtractor.hpp"
#include "imageprocessing/FilteringFeatureExtractor.hpp"
#include "imageprocessing/HistEq64Filter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"
#include "imageprocessing/ZeroMeanUnitVarianceFilter.hpp"
#include "imageprocessing/IntensityNormNormalizationFilter.hpp"
#include "imageprocessing/WhiteningFilter.hpp"

#include "detection/SlidingWindowDetector.hpp"
#include "detection/ClassifiedPatch.hpp"
#include "detection/OverlapElimination.hpp"
#include "detection/FiveStageSlidingWindowDetector.hpp"

#include "logging/LoggerFactory.hpp"
#include "imagelogging/ImageLoggerFactory.hpp"
#include "imagelogging/ImageFileWriter.hpp"

namespace po = boost::program_options;
using namespace std;
using namespace imageprocessing;
using namespace detection;
using namespace classification;
using namespace imageio;
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;
using imagelogging::ImageLogger;
using imagelogging::ImageLoggerFactory;
using boost::make_indirect_iterator;
using boost::property_tree::ptree;
using boost::property_tree::info_parser::read_info;
using boost::filesystem::path;
using boost::lexical_cast;


void drawBoxes(Mat image, vector<shared_ptr<ClassifiedPatch>> patches)
{
	for(const auto& cpatch : patches) {
		shared_ptr<Patch> patch = cpatch->getPatch();
		cv::rectangle(image, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((cpatch->getProbability())/1.0)   ));
	}
}

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
	return os;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	string verboseLevelImages;
	vector<path> inputPaths;
	path inputFilelist;
	path inputDirectory;
	vector<path> inputFilenames;
	path configFilename;
	shared_ptr<ImageSource> imageSource;
	path outputPicsDir;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("verbose-images,w", po::value<string>(&verboseLevelImages)->implicit_value("INTERMEDIATE")->default_value("FINAL","write images with FINAL loglevel or below."),
				  "specify the verbosity of the image output: FINAL, INTERMEDIATE, INFO, DEBUG or TRACE")
			("config,c", po::value<path>()->required(), 
				"path to a config (.cfg) file")
			("input,i", po::value<vector<path>>()->required(), 
				"input from one or more files, a directory, or a  .lst-file containing a list of images")
			("output-dir,o", po::value<path>()->default_value("."),
				"output directory for the result images")
		;

		po::positional_options_description p;
		p.add("input", -1);
		
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
		po::notify(vm);
	
		if (vm.count("help")) {
			cout << "Usage: mergePlaygroundApp [options]\n";
			cout << desc;
			return EXIT_SUCCESS;
		}
		if (vm.count("verbose")) {
			verboseLevelConsole = vm["verbose"].as<string>();
		}
		if (vm.count("verbose-images")) {
			verboseLevelImages = vm["verbose-images"].as<string>();
		}
		if (vm.count("input"))
		{
			inputPaths = vm["input"].as<vector<path>>();
		}
		if (vm.count("config"))
		{
			configFilename = vm["config"].as<path>();
		}
		if (vm.count("output-dir"))
		{
			outputPicsDir = vm["output-dir"].as<path>();
		}
	} catch(std::exception& e) {
		cout << e.what() << endl;
		return EXIT_FAILURE;
	}

	loglevel logLevel;
	if(boost::iequals(verboseLevelConsole, "PANIC")) logLevel = loglevel::PANIC;
	else if(boost::iequals(verboseLevelConsole, "ERROR")) logLevel = loglevel::ERROR;
	else if(boost::iequals(verboseLevelConsole, "WARN")) logLevel = loglevel::WARN;
	else if(boost::iequals(verboseLevelConsole, "INFO")) logLevel = loglevel::INFO;
	else if(boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = loglevel::DEBUG;
	else if(boost::iequals(verboseLevelConsole, "TRACE")) logLevel = loglevel::TRACE;
	else {
		cout << "Error: Invalid loglevel." << endl;
		return EXIT_FAILURE;
	}
	imagelogging::loglevel imageLogLevel;
	if(boost::iequals(verboseLevelImages, "FINAL")) imageLogLevel = imagelogging::loglevel::FINAL;
	else if(boost::iequals(verboseLevelImages, "INTERMEDIATE")) imageLogLevel = imagelogging::loglevel::INTERMEDIATE;
	else if(boost::iequals(verboseLevelImages, "INFO")) imageLogLevel = imagelogging::loglevel::INFO;
	else if(boost::iequals(verboseLevelImages, "DEBUG")) imageLogLevel = imagelogging::loglevel::DEBUG;
	else if(boost::iequals(verboseLevelImages, "TRACE")) imageLogLevel = imagelogging::loglevel::TRACE;
	else {
		cout << "Error: Invalid image loglevel." << endl;
		return EXIT_FAILURE;
	}
	
	Loggers->getLogger("classification").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("imageprocessing").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("detection").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("ffpDetectApp").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("ffpDetectApp");

	appLogger.debug("Verbose level for console output: " + logging::loglevelToString(logLevel));
	appLogger.debug("Verbose level for image output: " + imagelogging::loglevelToString(imageLogLevel));
	appLogger.debug("Using config: " + configFilename.string());
	appLogger.debug("Using output directory: " + outputPicsDir.string());
	if(outputPicsDir.empty()) {
		appLogger.info("Output directory not set. Writing images into current directory.");
	}

	ImageLoggers->getLogger("detection").addAppender(make_shared<imagelogging::ImageFileWriter>(imageLogLevel, outputPicsDir));
	ImageLoggers->getLogger("mergePlaygroundApp").addAppender(make_shared<imagelogging::ImageFileWriter>(imageLogLevel, outputPicsDir));


	
	// TEMP: Read a txt patchset from Cog, do WHI, and write a new .txt.
	// =================================================================
	vector<cv::Mat> positives;
	std::ifstream file("E:/training/ffd_training_regrPaperX200/data/posPatches.txt");
	if (!file.is_open()) {
		std::cout << "Invalid patches file" << std::endl;
		return 0;
	}
	while (file.good()) {
		string line;
		if (!std::getline(file, line))
			break;

		int width = 31;
		int height = 31;
		int dimensions = width * height;
		Mat patch(width, height, CV_8U);
		std::istringstream lineStream(line);
		if (!lineStream.good() || lineStream.fail()) {
			std::cout << "Invalid patches file l2" << std::endl;
			return 0;
		}
		uchar* values = patch.ptr<uchar>(0);
		float val;
		for (int j = 0; j < dimensions; ++j) {
			lineStream >> val;
			values[j] = static_cast<uchar>(val*255.0f);
		}

		positives.push_back(patch);
	}
	file.close();

	vector<cv::Mat> negatives;
	file.open("E:/training/ffd_training_regrPaperX200/data/negPatches.txt");
	if (!file.is_open()) {
		std::cout << "Invalid patches file" << std::endl;
		return 0;
	}
	while (file.good()) {
		string line;
		if (!std::getline(file, line))
			break;

		int width = 31;
		int height = 31;
		int dimensions = width * height;
		Mat patch(width, height, CV_8U);
		std::istringstream lineStream(line);
		if (!lineStream.good() || lineStream.fail()) {
			std::cout << "Invalid patches file l4" << std::endl;
			return 0;
		}
		uchar* values = patch.ptr<uchar>(0);
		float val;
		for (int j = 0; j < dimensions; ++j) {
			lineStream >> val;
			values[j] = static_cast<uchar>(val*255.0f);
		}

		negatives.push_back(patch);
	}
	file.close();

	shared_ptr<WhiteningFilter> fw = make_shared<WhiteningFilter>();
	shared_ptr<HistogramEqualizationFilter> fh = make_shared<HistogramEqualizationFilter>();
	shared_ptr<ConversionFilter> fc = make_shared<ConversionFilter>(CV_32F, 1.0/127.5, -1.0);
	shared_ptr<IntensityNormNormalizationFilter> fi = make_shared<IntensityNormNormalizationFilter>(cv::NORM_L2);
	shared_ptr<ReshapingFilter> fr = make_shared<ReshapingFilter>(1);
	for (auto& p : positives) {
		//p.convertTo(p, CV_32F);
		fw->applyInPlace(p);
		// min/max, stretch to [0, 255] 8U
		fh->applyInPlace(p);
		// need to go back to [-1, 1] before IntensityNormNormalizationFilter:
		fc->applyInPlace(p);
		fi->applyInPlace(p);
		fr->applyInPlace(p);
	}
	for (auto& p : negatives) {
		fw->applyInPlace(p);
		fh->applyInPlace(p);
		fc->applyInPlace(p);
		fi->applyInPlace(p);
		fr->applyInPlace(p);
	}

	std::ofstream ofile("E:/training/ffd_training_regrPaperX200/data/posPatchesWhi.txt");
	if (!ofile.is_open()) {
		std::cout << "Invalid patches file" << std::endl;
		return 0;
	}
	for (auto& p : positives) {
		int width = 31;
		int height = 31;
		int dimensions = width * height;
		float* values = p.ptr<float>(0);
		float val;
		for (int j = 0; j < dimensions; ++j) {
			ofile << values[j] << " ";
		}
		ofile << "\n";
	}
	ofile.close();

	ofile.open("E:/training/ffd_training_regrPaperX200/data/negPatchesWhi.txt");
	if (!ofile.is_open()) {
		std::cout << "Invalid patches file" << std::endl;
		return 0;
	}
	for (auto& p : negatives) {
		int width = 31;
		int height = 31;
		int dimensions = width * height;
		float* values = p.ptr<float>(0);
		float val;
		for (int j = 0; j < dimensions; ++j) {
			ofile << values[j] << " ";
		}
		ofile << "\n";
	}
	ofile.close();
	
	// END TEMP


	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	while(imageSource->next()) {

		img = imageSource->getImage();
		start = std::chrono::system_clock::now();

		// do something

		end = std::chrono::system_clock::now();

		int elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);

		stringstream ss;
		ss << std::ctime(&end_time);
		appLogger.info("finished computation at " + ss.str() + "elapsed time: " + lexical_cast<string>(elapsed_seconds) + "s or exactly " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");

	
	return 0;
}

