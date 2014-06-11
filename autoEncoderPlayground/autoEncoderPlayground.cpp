/*
 * autoEncoderPlayground.cpp
 *
 *  Created on: 11.06.2014
 *      Author: Patrik Huber
 *
 * Example:
 * -
 *   
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

#include <chrono>
#include <memory>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/lexical_cast.hpp"

#include "Eigen/Dense"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using cv::Mat;
using cv::Point2f;
using cv::Vec3f;
using cv::Scalar;
using boost::filesystem::path;
using boost::lexical_cast;
using std::cout;
using std::endl;
using std::make_shared;
using std::string;
using std::vector;

#include <random>
using cv::Rect;
std::vector<cv::Mat> generatePatches(std::vector<cv::Mat> images, int patchsize, int numPatches)
{
	if (images.size() == 0) {
		return vector<Mat>();
	}

	vector<Mat> patches;

	int imageWidth = images[0].cols;
	int imageHeight = images[0].rows;

	std::random_device rd; ///< Used for seeding the PRNG (mt19937)
	std::mt19937 engine(rd()); ///< A Mersenne twister mt19937 engine
	std::uniform_int_distribution<> randomImage(0, images.size() - 1);
	std::uniform_int_distribution<> randomPointX(0, imageWidth - patchsize - 1); // maybe -1 or 0 or +1?
	std::uniform_int_distribution<> randomPointY(0, imageHeight - patchsize -1);

	for (auto i = 0; i < numPatches; ++i) {
		int randomIndex = randomImage(engine);
		Rect patchRegion;
		patchRegion.x = randomPointX(engine);
		patchRegion.y = randomPointY(engine);
		patchRegion.width = patchsize;
		patchRegion.height = patchsize;
		patches.emplace_back(images[randomIndex](patchRegion).clone()); // Yep, we really want to deep copy the data here.
	}
	return patches;
}

std::vector<cv::Mat> normalizeData(std::vector<cv::Mat> features)
{
	if (features.size() == 0) {
		return vector<Mat>();
	}
	// Squash data to[0.1, 0.9] since we use sigmoid as the activation
	// function in the output layer

	// subtract mean from each patch individually:
	// Remove DC(mean of images).
	//patches = bsxfun(@minus, patches, mean(patches));
	for (auto&& f : features) {
		cv::Scalar mean = cv::mean(f);
		f.convertTo(f, CV_32FC1);
		f = f - mean; // it's all done in-place, data is never copied. Good!
	}

	// Truncate to + / -3 standard deviations and scale to - 1 to 1. stddev of ALL patches together.
	//pstd = 3 * std(patches(:));
	Mat allPatches(features[0].rows * features[0].cols, features.size(), CV_32FC1); // each column is 1 patch
	for (auto i = 0; i < features.size(); ++i) {
		Mat reshaped = features[i].reshape(1, features[0].rows * features[0].cols);
		reshaped.copyTo(allPatches.col(i));
	}
	cv::Scalar stddev;
	cv::Scalar mean;
	cv::meanStdDev(allPatches, mean, stddev);
	float pstd = 3 * stddev[0];
	//patches = max(min(patches, pstd), -pstd) / pstd; // % truncate each element individually
	allPatches = cv::max(cv::min(allPatches, pstd), -pstd) / pstd;
	// Rescale from[-1, 1] to[0.1, 0.9]
	allPatches = (allPatches + 1.0f) * 0.4f + 0.1f;

	vector<Mat> normalizedPatches;
	for (auto i = 0; i < allPatches.cols; ++i) {
		normalizedPatches.emplace_back(allPatches.col(i));
	}
	return normalizedPatches;
}

cv::Mat initializeParameters(int hiddenSize, int visibleSize)
{
	// Initialize parameters randomly based on layer sizes.
	float r = sqrt(6) / static_cast<float>(sqrt(hiddenSize + visibleSize + 1));   // we'll choose weights uniformly from the interval [-r, r]
	std::random_device rd; ///< Used for seeding the PRNG (mt19937)
	std::mt19937 engine(rd()); ///< A Mersenne twister mt19937 engine
	std::uniform_real_distribution<float> randomNumber(-r, r);
	
	Mat W1(hiddenSize, visibleSize, CV_32FC1);
	for (auto r = 0; r < W1.rows; ++r) {
		for (auto c = 0; c < W1.cols; ++c) {
			W1.at<float>(r, c) = randomNumber(engine);
		}
	}
	Mat W2(visibleSize, hiddenSize, CV_32FC1);
	for (auto r = 0; r < W2.rows; ++r) {
		for (auto c = 0; c < W2.cols; ++c) {
			W2.at<float>(r, c) = randomNumber(engine);
		}
	}

	Mat b1(hiddenSize, 1, CV_32FC1, 0.0f);
	Mat b2(visibleSize, 1, CV_32FC1, 0.0f);

	// Convert weights and bias gradients to the vector form.
	// This step will "unroll" (flatten and concatenate together) all
	// your parameters into a vector, which can then be used with minFunc.
	//theta = [W1(:); W2(:); b1(:); b2(:)];
	//Mat theta(W1.rows * W1.cols + W2.rows * W2.cols + b1.rows + b2.rows, 1, CV_32FC1);
	Mat theta;
	cv::vconcat(W1.reshape(1, hiddenSize * visibleSize), W2.reshape(1, hiddenSize * visibleSize), theta);
	cv::vconcat(theta, b1, theta);
	cv::vconcat(theta, b2, theta);
	return theta;
}

// visibleSize: the number of input units(probably 64)
// hiddenSize : the number of hidden units(probably 25)
// lambda : weight decay parameter
// sparsityParam : The desired average activation for the hidden units(denoted in the lecture
//                           notes by the greek alphabet rho, which looks like a lower - case "p").
// beta : weight of sparsity penalty term
// data : Our 64x10000 matrix containing the training data.So, data(:, i) is the i - th training example.

// The input theta is a vector(because minFunc expects the parameters to be a vector).
// We first convert theta to the(W1, W2, b1, b2) matrix / vector format, so that this
// follows the notation convention of the lecture notes.
// return: [cost, grad]
std::pair<float, float> sparseAutoencoderCost(cv::Mat theta, int visibleSize, int hiddenSize, float lambda, float sparsityParam, float beta, std::vector<cv::Mat> patches)
{
	return std::make_pair(0.0f, 0.0f);
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
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: autoEncoderPlayground [options]\n";
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
	Loggers->getLogger("autoEncoderPlayground").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("autoEncoderPlayground");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));
	
	path imagesroot(R"(C:\Users\Patrik\Documents\GitHub\FeatureDetection\autoEncoderPlayground\share\data\)");

	int visibleSize = 8 * 8;   // number of input units
	int hiddenSize = 25;     // number of hidden units
	float sparsityParam = 0.01f;   // desired average activation of the hidden units. rho
	float lambda = 0.0001f;     // weight decay parameter
	float beta = 3.0f;            // weight of sparsity penalty term

	vector<Mat> images;
	for (auto&& i = 1; i <= 10; ++i) {
		string imagepath = imagesroot.string() + string("image_") + std::to_string(i) + string(".png");
		images.emplace_back(cv::imread(imagepath, CV_LOAD_IMAGE_GRAYSCALE));
	}

	// randomly sample patches from the images, i.e. prepare the input to the AE training
	vector<Mat> features = generatePatches(images, 8, 10000); // input. 8x8 patches, 10000.
	features = normalizeData(features); // well, some strange normalization to [0.1, 0.9], mean/sdev normalization etc...

	Mat theta; ///< These are all our weights (including bias-terms) concatenated to a column-vector. 
	theta = initializeParameters(hiddenSize, visibleSize);
	// Note: I think we should make a class Parameter (CostParameter?...) that manages the en/un-rolling.
	// If we keep it like this, we need to keep track of hiddenSize etc.

	float cost, grad;
	std::tie(cost, grad) = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, features);
	return 0;
}
