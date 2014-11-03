/*
 * train-deeplearning-matcher-learn.cpp
 *
 *  Created on: 01.11.2014
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
	path trainingFramesFile, testFramesFile;
	path trainingSigsetFile, testSigsetFile;
	path outputPath;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("training-data,i", po::value<path>(&trainingFramesFile)->required(),
				"input file with training images (extracted frames) in boost::serialization text format")
			("training-sigset,j", po::value<path>(&trainingSigsetFile)->required(),
				"PaSC video query sigset, used for building the pairs and labels")
			("test-data,s", po::value<path>(&testFramesFile)->required(),
				"input file with test images (extracted frames) in boost::serialization text format")
			("test-sigset,t", po::value<path>(&testSigsetFile)->required(),
				"PaSC video query sigset, used for building the pairs and labels")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder to save the learned neural network")
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
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}

	// Prepare the training and test data, i.e. create the pairs:
	auto trainingQuerySigset = facerecognition::utils::readPascSigset(trainingSigsetFile, true);
	auto testQuerySigset = facerecognition::utils::readPascSigset(testSigsetFile, true);
	
	// The training data:
	vector<Mat> trainingFrames;
	{
		std::ifstream ifFrames(trainingFramesFile.string());
		boost::archive::text_iarchive ia(ifFrames);
		ia >> trainingFrames;
	}
	// The test data:
	vector<Mat> testFrames;
	{
		std::ifstream ifFrames(testFramesFile.string());
		boost::archive::text_iarchive ia(ifFrames);
		ia >> testFrames;
	}

	// Build the pairs, which will be the training data:
	vector<tiny_cnn::vec_t> trainingData; // preallocate?
	vector<tiny_cnn::label_t> trainingLabels; // preallocate?
	for (int q = 0; q < trainingQuerySigset.size(); ++q) {
		// Let's see that we only get the upper diagonal, or we'll have double pairs in it.
		// Okay actually we want that? It will be [a b] and [b a]
		for (int t = 0; t < trainingQuerySigset.size(); ++t) { // We might want to loop over the target sigset
			if (trainingFrames[q].empty() || trainingFrames[t].empty()) {
				continue;
			}
			Mat datum; // preallocate? No, because we don't want to actually copy the data
			cv::hconcat(trainingFrames[q], trainingFrames[t], datum); // Does this copy the data? We shouldn't!
			tiny_cnn::vec_t data; // vector<double>
			data.reserve(datum.cols);
			for (int i = 0; i < datum.cols; ++i) {
				data.emplace_back(datum.at<uchar>(0, i));
			}
			trainingData.emplace_back(data);
			if (trainingQuerySigset[q].subjectId == trainingQuerySigset[t].subjectId) {
				trainingLabels.emplace_back(1);
			}
			else {
				trainingLabels.emplace_back(0);
			}
			if (trainingData.size() >= 5000) {
				break;
			}
		}
	}

	// Build the pairs, which will be the test data:
	vector<tiny_cnn::vec_t> testData; // preallocate?
	vector<tiny_cnn::label_t> testLabels; // preallocate?
	for (int q = 0; q < testQuerySigset.size(); ++q) {
		// Let's see that we only get the upper diagonal, or we'll have double pairs in it.
		// Okay actually we want that? It will be [a b] and [b a]
		for (int t = 0; t < testQuerySigset.size(); ++t) { // We might want to loop over the target sigset
			if (testFrames[q].empty() || testFrames[t].empty()) {
				continue;
			}
			Mat datum; // preallocate? No, because we don't want to actually copy the data
			cv::hconcat(testFrames[q], testFrames[t], datum); // Does this copy the data? We shouldn't!
			tiny_cnn::vec_t data; // vector<double>
			data.reserve(datum.cols);
			for (int i = 0; i < datum.cols; ++i) {
				data.emplace_back(datum.at<uchar>(0, i));
			}
			testData.emplace_back(data);
			if (testQuerySigset[q].subjectId == testQuerySigset[t].subjectId) {
				testLabels.emplace_back(1);
			}
			else {
				testLabels.emplace_back(0);
			}
			if (testData.size() >= 150000) {
				break;
			}
		}
	}

	// Train NN:
	// MNIST: data is 28x28. On load, a border of 2 on each side gets added, resulting in 32x32. This is so that the convolution 5x5 window can reach each pixel as center.
	using namespace tiny_cnn;

	vector<label_t> train_labels, test_labels; // a int from 0 to 9 (MNIST)
	vector<vec_t> train_images, test_images; // double -1.0 to 1.0
	// vec_t is a std::vector<float_t>, float_t = double

	// scale the image data from [0, 255] to [-1.0, 1.0]. (later: subtract the mean as well?)
	/*	for (auto& f : trainingFrames) {
	Mat imageAsRowVector = f.reshape(1, 1);
	imageAsRowVector.convertTo(imageAsRowVector, CV_32FC1, 1.0 / 127.5, -1.0);
	train_images.emplace_back(vec_t(imageAsRowVector));
	}
	*/

	// Split whole data into train and test data:
/*	int numTrainingData = 7000;
	train_labels.assign(begin(trainingLabels), begin(trainingLabels) + numTrainingData);
	test_labels.assign(begin(trainingLabels) + numTrainingData, end(trainingLabels));
	train_images.assign(begin(trainingData), begin(trainingData) + numTrainingData);
	test_images.assign(begin(trainingData) + numTrainingData, end(trainingData));
*/
	train_images = trainingData;
	train_labels = trainingLabels;
	test_images = testData;
	test_labels = trainingLabels;

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

	std::cout << "start learning" << std::endl;

	boost::progress_display disp(train_images.size());
	boost::timer t;
	int minibatch_size = 100;

	nn.optimizer().alpha *= std::sqrt(minibatch_size);

	// create callback
	auto on_enumerate_epoch = [&](){
		std::cout << t.elapsed() << "s elapsed." << std::endl;

		tiny_cnn::result res = nn.test(test_images, test_labels);

		std::cout << "LR: " << nn.optimizer().alpha << ", numSuccess/numTotal: " << res.num_success << "/" << res.num_total << std::endl;

		nn.optimizer().alpha *= 0.85; // decay learning rate
		nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&](){
		disp += minibatch_size;

		// weight visualization in imdebug
		/*static int n = 0;
		n+=minibatch_size;
		if (n >= 1000) {
		image img;
		C3.weight_to_image(img);
		imdebug("lum b=8 w=%d h=%d %p", img.width(), img.height(), &img.data()[0]);
		n = 0;
		}*/
	};

	// training:
	// 20 = epochs. = For how many iterations to train the NN for. After each, we do testing.
	nn.train(train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	// test and show results
	nn.test(test_images, test_labels).print_detail(std::cout);

	// save networks
	std::ofstream ofs("LeNet-weights");
	ofs << C1 << S2 << C3 << S4 << C5 << F6;
	ofs.close();

	// Save it:

	// Test it:
	// - Error on train-set?
	// - Split the train set in train/test
	// - Error on 'real' PaSC part? (i.e. the validation set)

	return EXIT_SUCCESS;
}
