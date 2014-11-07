/*
 * cnn-matcher-extract-trainingdata.cpp
 *
 *  Created on: 01.11.2014
 *      Author: Patrik Huber
 *
 * Goal: Preprocessing.
 *
 * Notes: Maybe we need to change things because: We might want more positive pairs than negative.
 * So maybe we should build the pairs here? Maybe not, because we don't want to pre-build the test-pairs (2 Mio. and more).
 *
 * Also we build the mean etc over 512x1 only, not over the whole 1024x1 vector we shove into the CNN.
 *
 * Example:
 * ./cnn-matcher-extract-trainingdata -s "C:\Users\Patrik\Documents\GitHub\data\PaSC\Web\nd1Fall2010VideoPaSCTrainingSet.xml" -d "Z:\FRonPaSC\IJCB2014Competition\1_Preprocessing\SimpleFrameselect_training_video_best1_croppedHeads_affinealigned_Patrik_03112014" -o training_video_best1_cropped_aa.bs.txt
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
#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/archive/text_oarchive.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/utility.hpp"

#include "boost/timer.hpp"
#include "boost/progress.hpp"

#include "tiny_cnn.h"

#include "imageio/MatSerialization.hpp"

#include "facerecognition/utils.hpp"
#include "facerecognition/pasc.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using cv::Mat;
using boost::filesystem::path;
using boost::lexical_cast;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::pair;

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path sigsetFile;
	path inputPatchesDirectory;
	path outputFile;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("sigset,s", po::value<path>(&sigsetFile)->required(),
				  "PaSC video training sigset")
			("data,d", po::value<path>(&inputPatchesDirectory)->required(),
				"path to video frames, from sigset, cropped")
			("output,o", po::value<path>(&outputFile)->default_value("output.bs.txt"),
				"output file for the data, in boost::serialization text format. Folder needs to exist. The mean will be saved under the same filename with extension .mean.bs.txt.")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: cnn-matcher-extract-trainingdata [options]" << std::endl;
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
	
	Loggers->getLogger("imageio").addAppender(std::make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("app").addAppender(std::make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Create the output directory if it doesn't exist yet:
	//if (!boost::filesystem::exists(outputFile)) {
	//	boost::filesystem::create_directory(outputFile);
	//}
	
	auto sigset = facerecognition::utils::readPascSigset(sigsetFile, true);

	std::map<path, path> filesMap;
	{
		vector<path> files;
		try	{
			std::copy(fs::directory_iterator(inputPatchesDirectory), fs::directory_iterator(), std::back_inserter(files));
		}
		catch (const fs::filesystem_error& ex)
		{
			appLogger.error(ex.what());
			return EXIT_FAILURE;
		}
		// Convert to a map - otherwise, the search will be be very slow.
		for (auto& f : files) {
			filesMap.emplace(std::make_pair(f.stem().stem(), f));
		}
	}
	
	vector<string> subjectIds;
	Mat data;
	for (auto& s : sigset) {
		auto queryIter = filesMap.find(s.dataPath.stem());
		if (queryIter == end(filesMap)) {
			// No frame for this sigset entry. We just don't include it in our training data.
			continue;
		}
		// Todo/Note: What if we have several frames? Will the iterator just point to the first?
		Mat patch = cv::imread((inputPatchesDirectory / queryIter->second.filename()).string());
		// Here, we'd convert to gray, extract features etc.
		// Also we might read from the uncropped images, read in landmarks as well, and extract precise local features
		//cout << static_cast<float>(patch.rows) / patch.cols << endl;
		float aspect = 1.09f;
		cvtColor(patch, patch, cv::COLOR_RGB2GRAY);
		// (30, 33) would be better but because of tiny-cnn we choose 32x32 atm
		// No, we only have half the space, i.e. 1024 / 2 = 512 for one patch
		cv::resize(patch, patch, cv::Size(40, 44)); // could choose interpolation method
		patch = patch.reshape(1, 1); // Reshape to 1 row (row-vector - better suits OpenCV memory layout?)

		// Convert to float/double, scale to [-1.0, 1.0]
		patch.convertTo(patch, CV_64FC1, 1.0 / 127.5, -1.0);

		subjectIds.emplace_back(s.subjectId);
		data.push_back(patch);
	}
	// Calculate the mean
	Mat rowMean;
	cv::reduce(data, rowMean, 0, CV_REDUCE_AVG);
	// Subtract the mean. Maybe unit variance, save variance.
	data = data - cv::repeat(rowMean, data.rows, 1);

	// Convert to tiny-cnn data:
	std::vector<tiny_cnn::vec_t> train_data_original;
	for (int r = 0; r < data.rows; ++r) {
		tiny_cnn::vec_t datum; // vector<double>
		datum.reserve(data.cols);
		for (int i = 0; i < data.cols; ++i) {
			datum.emplace_back(data.at<double>(r, i));
		}
		train_data_original.emplace_back(datum);
	}

	// Maybe: add noise to the original data, and train with this as input?

	// Learn SDAE - reduce dim, learn better representation?
	//auto nn = tiny_cnn::make_mlp<tiny_cnn::mse, tiny_cnn::gradient_descent, tiny_cnn::activation::tan_h>({ 1760, 880, 440, 880, 1760 });
	using namespace tiny_cnn;
	typedef network<mse, gradient_descent> net_t;
	net_t nn;
	auto layer_sizes = { 1760, 500, 1760 };
	auto first = begin(layer_sizes);
	auto next = first + 1;
	auto last = end(layer_sizes);
	vector<layer_base<net_t>*> mlp_layers; // delete them before exiting?
	int cnt = 0;
	for (; next != last; ++first, ++next) {
		mlp_layers.emplace_back(new fully_connected_layer<net_t, activation::sigmoid>(*first, *next));
		nn.add(mlp_layers[cnt]);
		++cnt;
	}

	// create callbacks:
	int minibatch_size = 1;
	boost::progress_display disp(train_data_original.size());
	boost::timer t;
	auto on_enumerate_epoch = [&](){
		std::cout << t.elapsed() << "s elapsed." << std::endl;

		//std::cout << "LR: " << nn.optimizer().alpha << ", numSuccess/numTotal: " << res.num_success << "/" << res.num_total << std::endl;
		auto average_loss = nn.test(train_data_original);
		std::cout << "avg loss: per sample: " << average_loss << ", per pixel: " << average_loss / train_data_original[0].size() << ", per pixel (in 0-255): " << (average_loss / train_data_original[0].size()) * 127.5 << std::endl;

		nn.optimizer().alpha *= 0.85; // decay learning rate
		nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);

		disp.restart(train_data_original.size());
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
	// learning { 1760, 880, 440, 880, 1760 } denoising auto-encoder
	nn.train(train_data_original, train_data_original, minibatch_size, 5, on_enumerate_minibatch, on_enumerate_epoch);
	
	//std::ofstream nnOut("test.txt");
	//nnOut << mlp_layers[0];
	//nnOut.close();

	// Write out the original data and the reconstruction:
	int counter = 0;
	for (auto& in : train_data_original) {
		Mat inm(in);
		inm = inm.reshape(1, 44); // 40 (w=c) x 44 (h=r)
		inm.convertTo(inm, CV_8UC1, 127.5, 127.5);
		cv::imwrite("in" + facerecognition::getZeroPadded(counter) + ".png", inm);
		tiny_cnn::vec_t out;
		nn.predict(in, &out);
		Mat outm(out);
		outm = outm.reshape(1, 44);
		outm.convertTo(outm, CV_8UC1, 127.5, 127.5);
		cv::imwrite("out" + facerecognition::getZeroPadded(counter) + ".png", outm);
		++counter;
	}


	auto trainingData = std::make_pair(subjectIds, data);

	// Save the data:	
	{
		std::ofstream ofFrames(outputFile.string());
		boost::archive::text_oarchive oa(ofFrames);
		oa << trainingData;
	}

	// Save the mean:
	{
		path parent = outputFile.parent_path();
		path basename = outputFile.stem().stem();
		outputFile = parent / basename;
		outputFile.replace_extension(".mean.bs.txt");
		std::ofstream ofMean(outputFile.string());
		boost::archive::text_oarchive oa(ofMean);
		oa << rowMean;
	}

	appLogger.info("Finished preparing the training data. Saved data and the mean.");
	return EXIT_SUCCESS;
}
