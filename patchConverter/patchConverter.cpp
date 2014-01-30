/*
 * patchConverter.cpp
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
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"

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
#include "imageprocessing/UnitNormFilter.hpp"
#include "imageprocessing/WhiteningFilter.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
using namespace std;
using namespace imageprocessing;
using namespace imageio;
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;
using boost::property_tree::ptree;
using boost::property_tree::info_parser::read_info;
using boost::filesystem::path;
using boost::lexical_cast;


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
	path inputFilenamePos, inputFilenameNeg;
	path outputFilenamePos, outputFilenameNeg;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
				  ("input-positives,p", po::value<path>(&inputFilenamePos)->required(),
				"input file (.txt, containing positive patches)")
				("input-negatives,n", po::value<path>(&inputFilenameNeg)->required(),
				"input file (.txt, containing negative patches)")
				("output-positives,o", po::value<path>(&outputFilenamePos)->required(),
				"output directory for the positive result patches")
				("output-negatives,u", po::value<path>(&outputFilenameNeg)->required(),
				"output directory for the negative result patches")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		po::notify(vm);
	
		if (vm.count("help")) {
			cout << "Usage: patchConverter [options]\n";
			cout << desc;
			return EXIT_SUCCESS;
		}
		if (vm.count("verbose")) {
			verboseLevelConsole = vm["verbose"].as<string>();
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

	Loggers->getLogger("imageprocessing").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("patchConverter").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("patchConverter");

	appLogger.debug("Verbose level for console output: " + logging::loglevelToString(logLevel));
	
	// Read a txt patchset from Cog, do WHI, and write a new .txt.
	// =================================================================
	vector<cv::Mat> positives;
	std::ifstream file(inputFilenamePos.string()); // E:/training/ffd_training_regrPaperX200/data/posPatches.txt
	if (!file.is_open()) {
		appLogger.error("Error opening input file.");
		return EXIT_FAILURE;
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
	file.open(inputFilenameNeg.string()); // "E:/training/ffd_training_regrPaperX200/data/negPatches.txt"
	if (!file.is_open()) {
		appLogger.error("Error opening input file.");
		return EXIT_FAILURE;
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
	shared_ptr<UnitNormFilter> fi = make_shared<UnitNormFilter>(cv::NORM_L2);
	shared_ptr<ReshapingFilter> fr = make_shared<ReshapingFilter>(1);
	for (auto& p : positives) {
		//p.convertTo(p, CV_32F);
		fw->applyInPlace(p);
		// min/max, stretch to [0, 255] 8U
		fh->applyInPlace(p);
		// need to go back to [-1, 1] before UnitNormFilter:
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
	
	std::ofstream ofile(outputFilenamePos.string()); // "E:/training/ffd_training_regrPaperX200/data/posPatchesWhi.txt"
	if (!ofile.is_open()) {
		appLogger.error("Error creating output file.");
		return EXIT_FAILURE;
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

	ofile.open(outputFilenameNeg.string()); // "E:/training/ffd_training_regrPaperX200/data/negPatchesWhi.txt"
	if (!ofile.is_open()) {
		appLogger.error("Error creating output file.");
		return EXIT_FAILURE;
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
	
	return 0;
}

