/*
 * train-frame-extract-nnet.cpp
 *
 *  Created on: 11.09.2014
 *      Author: Patrik Huber
 *
 * Ideally we'd use video, match against highres stills? (and not the lowres). Because if still are lowres/bad, we could match a
 * good frame against a bad gallery, which would give a bad score, but it shouldn't, because the frame is good.
 * Do we have labels for this?
 * Maybe "sensor_id","stage_id","env_id","illuminant_id" in the files emailed by Ross.
 *
 * Example:
 * train-frame-extract-nnet ...
 *   
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>

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

#include <boost/timer.hpp>
#include <boost/progress.hpp>

#include "tiny_cnn.h"

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
	path inputFrames;
	path inputLabels;
	path outputPath;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("images,i", po::value<path>(&inputFrames)->required(),
				"input file with training images (extracted frames) in boost::serialization text format")
			("labels,l", po::value<path>(&inputLabels)->required(),
				"input labels for the training images, in boost::serialization text format")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
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
	Loggers->getLogger("train-frame-extract-nnet").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("train-frame-extract-nnet");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Create the output directory if it doesn't exist yet:
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	

	
	
	// The training data:
	vector<Mat> trainingFrames;
	vector<float> labels; // the score difference to the value we would optimally like
						  // I.e. if it's a positive pair, the label is the difference to 1.0
						  // In case of a negative pair, the label is the difference to 0.0



	// Train NN:
	// trainingFrames, labels
	// MNIST: data is 28x28. On load, a border of 2 on each side gets added, resulting in 32x32. This is so that the convolution 5x5 window can reach each pixel as center.
	using namespace tiny_cnn;

	std::vector<label_t> train_labels, test_labels; // a int from 0 to 9 (MNIST)
	std::vector<vec_t> train_images, test_images; // double -1.0 to 1.0
	// vec_t is a std::vector<float_t>, float_t = double

	train_labels = labels;
	test_labels = labels;
	for (auto& f : trainingFrames) {
		Mat imageAsRowVector = f.reshape(1, 1);
		vec_t test(imageAsRowVector);
		cout << "hi!";
	}

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
	fully_connected_layer<CNN, activation::tan_h> F6(120, 1); // int in_dim, int out_dim

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
	int minibatch_size = 10;

	nn.optimizer().alpha *= std::sqrt(minibatch_size);

	// create callback
	auto on_enumerate_epoch = [&](){
		std::cout << t.elapsed() << "s elapsed." << std::endl;

		tiny_cnn::result res = nn.test(test_images, test_labels);

		std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

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

	// training
	nn.train(train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	// test and show results
	nn.test(test_images, test_labels).print_detail(std::cout);

	// save networks
	std::ofstream ofs("LeNet-weights");
	ofs << C1 << S2 << C3 << S4 << C5 << F6;

	// Save it:

	// Test it:
	// - Error on train-set?
	// - Split the train set in train/test
	// - Error on 'real' PaSC part? (i.e. the validation set)

	return EXIT_SUCCESS;
}
