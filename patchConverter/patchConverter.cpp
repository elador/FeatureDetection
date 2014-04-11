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
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/filesystem/path.hpp"

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/ReshapingFilter.hpp"
#include "imageprocessing/ConversionFilter.hpp"
#include "imageprocessing/ResizingFilter.hpp"
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
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;
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
	path inputFilename;
	path outputFilename;
	int patchWidth, patchHeight;
	enum class ConversionMethod {
		H,
		WHI
	};
	int conversionMethodNum;
	ConversionMethod conversionMethod;
	bool doResize;
	int resizedWidth, resizedHeight;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("input,i", po::value<path>(&inputFilename)->required(),
				"input file (.txt, containing patches)")
			("output,o", po::value<path>(&outputFilename)->required(),
				"output file for the result patches")
			("patch-width,w", po::value<int>(&patchWidth)->required(),
				"output file for the result patches")
			("patch-height,h", po::value<int>(&patchHeight)->required(),
				"output file for the result patches")
			("method,m", po::value<int>(&conversionMethodNum)->required(),
				"conversion method: 0 = H, 1 = WHI")
			("resize,r", po::value<bool>(&doResize)->default_value(false)->implicit_value(true),
				"todo")
			("resized-width,s", po::value<int>(&resizedWidth),
				"todo")
			("resized-height,t", po::value<int>(&resizedHeight),
				"todo")
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
		cout << "Invalid loglevel." << endl;
		return EXIT_FAILURE;
	}

	Loggers->getLogger("imageprocessing").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("patchConverter").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("patchConverter");

	appLogger.debug("Verbose level for console output: " + logging::loglevelToString(logLevel));

	if (conversionMethodNum == 0) {
		conversionMethod = ConversionMethod::H;
	}
	else if (conversionMethodNum == 1) {
		conversionMethod = ConversionMethod::WHI;
	} else {
		appLogger.error("Unknown conversion method.");
		return EXIT_FAILURE;
	}
	
	// Read a txt patchset from Cog, do WHI, and write a new .txt.
	// =================================================================
	// -i C:\Users\Patrik\Documents\GitHub\tmp_regre_redMachines\data\posPatches.txt
	// -i C:\Users\Patrik\Documents\GitHub\tmp_regre_redMachines\data\negPatches.txt
	// -w 31 -h 31
	// Example usage:
	// -i C:\Users\Patrik\Documents\GitHub\tmp_regre_redMachines\data\posPatches.txt -o C:\Users\Patrik\Documents\GitHub\tmp_regre_redMachines\data\posPatches_out.txt -w 31 -h 31 -m 0 -r -s 19 -t 19
	vector<cv::Mat> patches;
	std::ifstream file(inputFilename.string());
	if (!file.is_open()) {
		appLogger.error("Error opening input file.");
		return EXIT_FAILURE;
	}
	while (file.good()) {
		string line;
		if (!std::getline(file, line))
			break;

		int dimensions = patchWidth * patchHeight;
		Mat patch(patchHeight, patchWidth, CV_8U);
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

		patches.push_back(patch);
	}
	file.close();

	vector<shared_ptr<ImageFilter>> filters;

	if (doResize) {
		filters.push_back(make_shared<ResizingFilter>(cv::Size(resizedWidth, resizedHeight)));
	}

	if (conversionMethod == ConversionMethod::H) {
		// input is [0, 1] float, we convert to [0, 255] on loading because HistEq needs U8
		filters.push_back(make_shared<HistogramEqualizationFilter>()); // min/max, stretch to [0, 255] 8U
		filters.push_back(make_shared<ConversionFilter>(CV_32F, 1.0 / 127.5, -1.0)); // go back to [-1, 1]
		filters.push_back(make_shared<ReshapingFilter>(1));
	} 
	else if (conversionMethod == ConversionMethod::WHI) {
		filters.push_back(make_shared<WhiteningFilter>());
		filters.push_back(make_shared<HistogramEqualizationFilter>()); // min/max, stretch to [0, 255] 8U
		filters.push_back(make_shared<ConversionFilter>(CV_32F, 1.0 / 127.5, -1.0)); // need to go back to [-1, 1] before UnitNormFilter
		filters.push_back(make_shared<UnitNormFilter>(cv::NORM_L2));
		filters.push_back(make_shared<ReshapingFilter>(1));
	}

	for (auto& p : patches) {
		for (const auto& f : filters) {
			f->applyInPlace(p);
		}
	}

	if (doResize) {
		patchWidth = resizedWidth;
		patchHeight = resizedHeight;
	}
	
	std::ofstream ofile(outputFilename.string());
	if (!ofile.is_open()) {
		appLogger.error("Error creating output file.");
		return EXIT_FAILURE;
	}
	for (auto& p : patches) {
		int dimensions = patchWidth * patchHeight;
		float* values = p.ptr<float>(0);
		for (int j = 0; j < dimensions; ++j) {
			ofile << values[j] << " ";
		}
		ofile << "\n";
	}
	ofile.close();
	
	return 0;
}

