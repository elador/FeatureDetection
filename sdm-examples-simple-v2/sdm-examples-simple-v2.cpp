/*
 * sdm-train-v2.cpp
 *
 *  Created on: 15.11.2014
 *      Author: Patrik Huber
 */

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
//#define _CRTDBG_MAP_ALLOC
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

#include <memory>
#include <iostream>
#include <cmath>

#include "opencv2/core/core.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/math/special_functions/erf.hpp"

#include "superviseddescent/superviseddescent.hpp" // v2 stuff

#include "logging/LoggerFactory.hpp"

using namespace superviseddescent;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using cv::Mat;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;

template<typename ForwardIterator, typename T>
void strided_iota(ForwardIterator first, ForwardIterator last, T value, T stride)
{
	while (first != last) {
		*first++ = value;
		value += stride;
	}
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: sdm-examples-simple-v2 [options]" << std::endl;
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
	
	Loggers->getLogger("superviseddescent").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// $\mathbf{h}$ generic, should return a cv::Mat row-vector (1 row). Should process 1 training data. Optimally, whatever it is - float or a whole cv::Mat. But we can also simulate the 1D-case with a 1x1 cv::Mat for now.
	// sin(x):
/*	auto h = [](float value) { return std::sin(value); };
	auto h_inv = [](float value) {
		if (value >= 1.0f) // our upper border of y is 1.0f, but it can be a bit larger due to floating point representation. asin then returns NaN.
			return std::asin(1.0f);
		else
			return std::asin(value);
		};
	// x^3:
	auto h = [](float value) { return std::pow(value, 3); };
	auto h_inv = [](float value) { return std::cbrt(value); }; // cubic root
	// erf(x):
	auto h = [](float value) { return std::erf(value); };
	auto h_inv = [](float value) { return boost::math::erf_inv(value); };
*/	// exp(x):
	auto h = [](float value) { return std::exp(value); };
	auto h_inv = [](float value) { return std::log(value); };

	//float startInterval = -1.0f; float stepSize = 0.2f; int numValues = 11; Mat y_tr(numValues, 1, CV_32FC1); // sin: [-1:0.2:1]
	//float startInterval = -27.0f; float stepSize = 3.0f; int numValues = 19; Mat y_tr(numValues, 1, CV_32FC1); // cube: [-27:3:27]
	//float startInterval = -0.99f; float stepSize = 0.11f; int numValues = 19; Mat y_tr(numValues, 1, CV_32FC1); // erf: [-0.99:0.11:0.99]
	float startInterval = 1.0f; float stepSize = 3.0f; int numValues = 10; Mat y_tr(numValues, 1, CV_32FC1); // exp: [1:3:28]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	//v2::SupervisedDescentOptimiser sdo({ make_shared<v2::LinearRegressor>(v2::LinearRegressor::RegularisationType::Manual, 0.0f) });
	vector<v2::LinearRegressor> regressors;
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	v2::SupervisedDescentOptimiser<v2::LinearRegressor> sdo(regressors);
	sdo.train(x_tr, y_tr, x0, h);
	
	// Test the trained model:
	// Test data with finer resolution:
	//float startIntervalTest = -1.0f; float stepSizeTest = 0.05f; int numValuesTest = 41; Mat y_ts(numValuesTest, 1, CV_32FC1); // sin: [-1:0.05:1]
	//float startIntervalTest = -27.0f; float stepSizeTest = 0.5f; int numValuesTest = 109; Mat y_ts(numValuesTest, 1, CV_32FC1); // cube: [-27:0.5:27]
	//float startIntervalTest = -0.99f; float stepSizeTest = 0.03f; int numValuesTest = 67; Mat y_ts(numValuesTest, 1, CV_32FC1); // erf: [-0.99:0.03:0.99]
	float startIntervalTest = 1.0f; float stepSizeTest = 0.5f; int numValuesTest = 55; Mat y_ts(numValuesTest, 1, CV_32FC1); // exp: [1:0.5:28]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.
	sdo.test(x_ts_gt, y_ts, x0_ts, h); // x_ts_gt will only be used to calculate the residual
	
	appLogger.info("Finished. Exiting...");

	return 0;
}
