/*
 * train-deeplearning-matcher-match.cpp
 *
 *  Created on: 06.11.2014
 *      Author: Patrik Huber
  *
 * Example:
 * train-deeplearning-matcher ...
 *   
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/serialization/vector.hpp"

#include <boost/timer.hpp>
#include <boost/progress.hpp>

#include "tiny_cnn.h"

#include "imageio/MatSerialization.hpp"

#include "facerecognition/utils.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using cv::Mat;
using boost::filesystem::path;
using boost::lexical_cast;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::vector;
using std::string;

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path cnnWeightsFile, testDataFile, testSigsetFile;
	path outputFile;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("weights,w", po::value<path>(&cnnWeightsFile)->required(),
				"weights of a trained tiny-cnn")
			("data,d", po::value<path>(&testDataFile)->required(),
				"input file with test images (extracted frames) in boost::serialization text format")
			("sigset,s", po::value<path>(&testSigsetFile)->required(),
				"PaSC video sigset")
			("output,o", po::value<path>(&outputFile)->default_value("output.csv"),
				"path to an output folder to save the simi mtx csv")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: train-frameselect-learn [options]" << std::endl;
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
	
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Create the output directory if it doesn't exist yet:
	if (!boost::filesystem::exists(outputFile)) {
		boost::filesystem::create_directory(outputFile);
	}

	// Prepare the training and test data, i.e. create the pairs:
	auto videoSigset = facerecognition::utils::readPascSigset(testSigsetFile, true);
	
	// The training data:
	vector<Mat> testData; // 1401 entries, same order as sigset. Mat is empty Mat() when image couldn't be enroled.
	{
		std::ifstream ifFrames(testDataFile.string());
		boost::archive::text_iarchive ia(ifFrames);
		ia >> testData;
	}

	// NN definition:
	// MNIST: data is 28x28. On load, a border of 2 on each side gets added, resulting in 32x32. This is so that the convolution 5x5 window can reach each pixel as center.
	using namespace tiny_cnn;
	typedef network<mse, gradient_descent> CNN;
	CNN nn;
	// First layer:
	int in_width = 32;
	int in_height = 32;
	int window_size = 5;
	int in_channels = 1;
	int out_channels = 6;
	// in: 32x32x1, out: 28x28x6
	// param_size: 150 w + 6 b; 5*5 window = 25, 25 * 6 chan = 150. Plus 1 bias for each chan = 156.
	// connection_size: 150 * 784 + 6 * 784; (28*28=784)
	convolutional_layer<CNN, activation::rectified_linear> C1(in_width, in_height, window_size, in_channels, out_channels);
	// in: 28x28x6, out: 14x14x6
	average_pooling_layer<CNN, activation::rectified_linear> S2(28, 28, 6, 2); // int in_width, int in_height, int in_channels, int pooling_size

	// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
	static const bool connection[] = {
		O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
		O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
		O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
		X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
		X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
		X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
	};
#undef O
#undef X
	// in: 14x14x6, out: 10x10x16
	convolutional_layer<CNN, activation::tan_h> C3(14, 14, 5, 6, 16, connection_table(connection, 6, 16)); // in_width, in_height, window_size, in_channels, out_channels
	// in: 10x10x16, out: 5x5x16
	average_pooling_layer<CNN, activation::tan_h> S4(10, 10, 16, 2);
	// in: 5x5x16, out: 1x1x128
	convolutional_layer<CNN, activation::tan_h> C5(5, 5, 5, 16, 120);
	// in: 1x1x128, out: 10?
	fully_connected_layer<CNN, activation::tan_h> F6(120, 2); // int in_dim, int out_dim

	assert(C1.param_size() == 156 && C1.connection_size() == 122304);
	assert(S2.param_size() == 12 && S2.connection_size() == 5880);
	assert(C3.param_size() == 1516 && C3.connection_size() == 151600); // 100 weight-dim, 2400 + 16 = 2416
	assert(S4.param_size() == 32 && S4.connection_size() == 2000);
	assert(C5.param_size() == 48120 && C5.connection_size() == 48120);

	nn.add(&C1);
	nn.add(&S2);
	nn.add(&C3);
	nn.add(&S4);
	nn.add(&C5);
	nn.add(&F6);

	// Load stored network weights:
	std::ifstream ifs(cnnWeightsFile.string());
	ifs >> C1 >> S2 >> C3 >> S4 >> C5 >> F6;
	ifs.close();

	Mat fullSimilarityMatrix = Mat::zeros(videoSigset.size(), videoSigset.size(), CV_32FC1);
	// Later on, targetFirFiles and queryFirFiles could be empty if no frame of a video could be
	// enroled. We'll just skip over them and leave these entries 0.

	for (size_t q = 0; q < videoSigset.size(); ++q)
	{
		appLogger.info("Matching query video " + std::to_string(q + 1) + " of " + std::to_string(videoSigset.size()) + "...");
		for (size_t t = q; t < videoSigset.size(); ++t) // start at t = q to only match the upper diagonal (scores are symmetric). Actually no. Maybe if we train also with symmetric pairs.
		{
			if (testData[q].empty() || testData[t].empty()) { // q could go into outer loop
				continue;
			}
			// Match the query frame against the target frame:
			Mat datum; // preallocate? No, because we don't want to actually copy the data
			cv::hconcat(testData[q], testData[t], datum); // Does this copy the data? We shouldn't!
			tiny_cnn::vec_t data; // vector<double>
			data.reserve(datum.cols);
			for (int i = 0; i < datum.cols; ++i) {
				data.emplace_back(datum.at<uchar>(0, i));
			}
			// Predict:
			tiny_cnn::vec_t out;
			nn.predict(data, &out);
			float score;
			// class 0 = not the same, class 1 = the same individual
			if (out[0] >= out[1]) { // It's class 0, i.e. not the same
				score = out[0];
			}
			else {
				score = out[1];
			}
			// Save:
			fullSimilarityMatrix.at<float>(q, t) = score;
			fullSimilarityMatrix.at<float>(t, q) = score; // fill the lower diagonal - assume scores are symmetric, which they're probably not. But for now.
		} // end for over all target videos
	} // end for over all query videos

	appLogger.info("Finished filling the full similarity matrix. Saving as CSV...");
	facerecognition::utils::saveSimilarityMatrixAsCSV(fullSimilarityMatrix, outputFile);
	appLogger.info("Successfully saved PaSC CSV similarity matrix.");

	return EXIT_SUCCESS;
}
