/*
 * FaceTracking.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "FaceTracking.h"
#include "imageio/VideoImageSource.h"
#include "imageio/KinectImageSource.h"
#include "imageio/DirectoryImageSource.h"
#include "classification/FrameBasedSvmTraining.h"
#include "classification/FastSvmTraining.h"
#include "classification/LibSvmParameterBuilder.h"
#include "classification/RbfLibSvmParameterBuilder.h"
#include "classification/PolyLibSvmParameterBuilder.h"
#include "classification/SigmoidParameterComputation.h"
#include "classification/ApproximateSigmoidParameterComputation.h"
#include "classification/FixedApproximateSigmoidParameterComputation.h"
#include "classification/TrainableClassifier.h"
#include "classification/LibSvmClassifier.h"
#include "classification/HistEqFeatureExtractor.h"
#include "tracking/ResamplingSampler.h"
#include "tracking/GridSampler.h"
#include "tracking/LowVarianceSampling.h"
#include "tracking/SimpleTransitionModel.h"
#include "tracking/WvmSvmModel.h"
#include "tracking/SelfLearningMeasurementModel.h"
#include "tracking/PositionDependentMeasurementModel.h"
#include "tracking/FilteringPositionExtractor.h"
#include "tracking/WeightedMeanPositionExtractor.h"
#include "tracking/Rectangle.h"
#include "tracking/Sample.h"
#include "OverlapElimination.h"
#include "VDetectorVectorMachine.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "SLogger.h"
#include <boost/optional.hpp>
#include <boost/program_options.hpp>
#include <vector>
#include <iostream>
#ifdef WIN32
	#include "wingettimeofday.h"
#else
	#include <sys/time.h>
#endif

namespace po = boost::program_options;

const std::string FaceTracking::videoWindowName = "Image";
const std::string FaceTracking::controlWindowName = "Controls";

FaceTracking::FaceTracking(auto_ptr<imageio::ImageSource> imageSource,
		std::string svmConfigFile, std::string negativesFile) :
				svmConfigFile(svmConfigFile), negativesFile(negativesFile), imageSource(imageSource) {
	initTracking();
	initGui();
}

FaceTracking::~FaceTracking() {}

void FaceTracking::initTracking() {
	// create static measurement model
	shared_ptr<VDetectorVectorMachine> wvm = make_shared<DetectorWVM>();
	wvm->load(svmConfigFile);
	shared_ptr<VDetectorVectorMachine> svm = make_shared<DetectorSVM>();
	svm->load(svmConfigFile);
	shared_ptr<OverlapElimination> oe = make_shared<OverlapElimination>();
	oe->load(svmConfigFile);
	staticMeasurementModel = make_shared<WvmSvmModel>(wvm, svm, oe);

	// create adaptive measurement model
	shared_ptr<LibSvmParameterBuilder> svmParameterBuilder = make_shared<RbfLibSvmParameterBuilder>(0.05);
//	shared_ptr<LibSvmParameterBuilder> svmParameterBuilder = make_shared<PolyLibSvmParameterBuilder>(2, 1, 1);

	shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation = make_shared<FixedApproximateSigmoidParameterComputation>();
//	shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation = make_shared<FastApproximateSigmoidParameterComputation>();
//	shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation = make_shared<ApproximateSigmoidParameterComputation>();

//	shared_ptr<LibSvmTraining> training = make_shared<FastSvmTraining>(7, 14, 50, svmParameterBuilder, sigmoidParameterComputation);
	shared_ptr<LibSvmTraining> training = make_shared<FrameBasedSvmTraining>(5, 4, svmParameterBuilder, sigmoidParameterComputation);
//	training->readStaticNegatives(negativesFile, 200);
	shared_ptr<TrainableClassifier> classifier = make_shared<LibSvmClassifier>(training);

	shared_ptr<HistEqFeatureExtractor> featureExtractor = make_shared<HistEqFeatureExtractor>(cv::Size(20, 20), 0.85, 0.1666, 1.0);

//	adaptiveMeasurementModel = make_shared<SelfLearningMeasurementModel>(featureExtractor, dynamicSvm, 0.85, 0.05);
	adaptiveMeasurementModel = make_shared<PositionDependentMeasurementModel>(featureExtractor, classifier);

	// create tracker
	unsigned int count = 800;
	double randomRate = 0.35;
	transitionModel = make_shared<SimpleTransitionModel>(0.2);
	resamplingSampler = make_shared<ResamplingSampler>(count, randomRate, make_shared<LowVarianceSampling>(),
			transitionModel, 0.1666, 1.0);
	gridSampler = make_shared<GridSampler>(0.1666, 0.8, 1 / 0.85, 0.1);
	tracker = auto_ptr<AdaptiveCondensationTracker>(new AdaptiveCondensationTracker(
			resamplingSampler, staticMeasurementModel, adaptiveMeasurementModel,
			make_shared<FilteringPositionExtractor>(make_shared<WeightedMeanPositionExtractor>())));
//	tracker->setUseAdaptiveModel(false);
}

void FaceTracking::initGui() {
	drawSamples = true;

	cvNamedWindow(controlWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(controlWindowName.c_str(), 750, 50);

	cv::createTrackbar("Adaptive", controlWindowName, NULL, 1, adaptiveChanged, this);
	cv::setTrackbarPos("Adaptive", controlWindowName, tracker->isUsingAdaptiveModel() ? 1 : 0);

	cv::createTrackbar("Grid/Resampling", controlWindowName, NULL, 1, samplerChanged, this);
	cv::setTrackbarPos("Grid/Resampling", controlWindowName, tracker->getSampler() == gridSampler ? 0 : 1);

	cv::createTrackbar("Sample Count", controlWindowName, NULL, 2000, sampleCountChanged, this);
	cv::setTrackbarPos("Sample Count", controlWindowName, resamplingSampler->getCount());

	cv::createTrackbar("Random Rate", controlWindowName, NULL, 100, randomRateChanged, this);
	cv::setTrackbarPos("Random Rate", controlWindowName, 100 * resamplingSampler->getRandomRate());

	cv::createTrackbar("Scatter * 100", controlWindowName, NULL, 100, scatterChanged, this);
	cv::setTrackbarPos("Scatter * 100", controlWindowName, 100 * transitionModel->getScatter());

	cv::createTrackbar("Draw samples", controlWindowName, NULL, 1, drawSamplesChanged, this);
	cv::setTrackbarPos("Draw samples", controlWindowName, drawSamples ? 1 : 0);
}

void FaceTracking::adaptiveChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->tracker->setUseAdaptiveModel(state == 1);
}

void FaceTracking::samplerChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	if (state == 0)
		tracking->tracker->setSampler(tracking->gridSampler);
	else
		tracking->tracker->setSampler(tracking->resamplingSampler);
}

void FaceTracking::sampleCountChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->resamplingSampler->setCount(state);
}

void FaceTracking::randomRateChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->resamplingSampler->setRandomRate(0.01 * state);
}

void FaceTracking::scatterChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->transitionModel->setScatter(0.01 * state);
}

void FaceTracking::drawSamplesChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->drawSamples = (state == 1);
}

void FaceTracking::drawDebug(cv::Mat& image) {
	cv::Scalar black(0, 0, 0); // blue, green, red
	cv::Scalar red(0, 0, 255); // blue, green, red
	cv::Scalar green(0, 255, 0); // blue, green, red
	if (drawSamples) {
		const std::vector<Sample> samples = tracker->getSamples();
		for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
			cv::Scalar color = sit->isObject() ? cv::Scalar(0, 0, sit->getWeight() * 255) : black;
			cv::circle(image, cv::Point(sit->getX(), sit->getY()), 3, color);
		}
	}
	cv::Scalar& svmIndicatorColor = tracker->wasUsingAdaptiveModel() ? green : red;
	cv::circle(image, cv::Point(10, 10), 5, svmIndicatorColor, -1);
}

void FaceTracking::run() {
	running = true;
	paused = false;

	bool first = true;
	cv::Mat frame, image;
	cv::Scalar green(0, 255, 0); // blue, green, red
	cv::Scalar red(0, 0, 255); // blue, green, red

	timeval start, detStart, detEnd, frameStart, frameEnd;
	float allIterationTimeSeconds = 0, allDetectionTimeSeconds = 0;

	int frames = 0;
	gettimeofday(&start, 0);
	std::cout.precision(2);

	while (running) {
		frames++;
		gettimeofday(&frameStart, 0);
		frame = imageSource->get();

		if (frame.empty()) {
			std::cerr << "Could not capture frame - press 'q' to quit program" << std::endl;
			stop();
			while ('q' != (char)cv::waitKey(10));
		} else {
			if (first) {
				first = false;
				image.create(frame.rows, frame.cols, frame.type());
			}
			gettimeofday(&detStart, 0);
			boost::optional<tracking::Rectangle> face = tracker->process(frame);
			gettimeofday(&detEnd, 0);
			image = frame;
			drawDebug(image);
			cv::Scalar& color = tracker->wasUsingAdaptiveModel() ? green : red;
			if (face)
				cv::rectangle(image, cv::Point(face->getX(), face->getY()),
						cv::Point(face->getX() + face->getWidth(), face->getY() + face->getHeight()), color);
			imshow(videoWindowName, image);
			gettimeofday(&frameEnd, 0);

			int iterationTimeMilliseconds = 1000 * (frameEnd.tv_sec - frameStart.tv_sec) + (frameEnd.tv_usec - frameStart.tv_usec) / 1000;
			int detectionTimeMilliseconds = 1000 * (detEnd.tv_sec - detStart.tv_sec) + (detEnd.tv_usec - detStart.tv_usec) / 1000;
			allIterationTimeSeconds += 0.001 * iterationTimeMilliseconds;
			allDetectionTimeSeconds += 0.001 * detectionTimeMilliseconds;
			float iterationFps = frames / allIterationTimeSeconds;
			float detectionFps = frames / allDetectionTimeSeconds;
			std::cout << "frame: " << frames << "; time: "
					<< iterationTimeMilliseconds << " ms (" << iterationFps << " fps); detection: "
					<< detectionTimeMilliseconds << " ms (" << detectionFps << " fps)" << std::endl;

			int delay = paused ? 0 : 5;
			char c = (char)cv::waitKey(delay);
			if (c == 'p')
				paused = !paused;
			else if (c == 'q')
				stop();
		}
	}
}

void FaceTracking::stop() {
	running = false;
}

int main(int argc, char *argv[]) {

	int verboseLevelText;
	int verboseLevelImages;
	int deviceId, kinectId;
	std::string filename, directory;
	bool useCamera = false, useKinect = false, useFile = false, useDirectory = false;
	std::string svmConfigFile, negativeRtlPatches;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "Produce help message")
			("verbose-text,v", po::value<int>(&verboseLevelText)->implicit_value(2)->default_value(0,"minimal text output"), "Enable text-verbosity (optionally specify level)")
			("verbose-images,w", po::value<int>(&verboseLevelImages)->implicit_value(2)->default_value(0,"minimal image output"), "Enable image-verbosity (optionally specify level)")
			("filename,f", po::value< std::string >(&filename), "A filename of a video to run the tracking")
			("directory,i", po::value< std::string >(&directory), "Use a directory as input")
			("device,d", po::value<int>(&deviceId)->implicit_value(0), "A camera device ID for use with the OpenCV camera driver")
			("kinect,k", po::value<int>(&kinectId)->implicit_value(0), "Windows only: Use a Kinect as camera. Optionally specify a device ID.")
			("config,c", po::value< std::string >(&svmConfigFile)->default_value("fd_config_fft_fd.mat","fd_config_fft_fd.mat"), "The filename to the config file that contains the SVM and WVM classifiers.")
			("nonfaces,n", po::value< std::string >(&negativeRtlPatches)->default_value("nonfaces_1000","nonfaces_1000"), "Filename to a file containing the static negative training examples for the real-time learning SVM.")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
			std::cout << "Usage: faceTrackingApp [options]" << std::endl;
			std::cout << desc;
			return 0;
		}
		if (vm.count("filename"))
			useFile = true;
		if (vm.count("directory"))
			useDirectory = true;
		if (vm.count("device"))
			useCamera = true;
		if (vm.count("kinect"))
			useKinect = true;
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return -1;
	}

	int inputsSpecified = 0;
	if (useCamera)
		inputsSpecified++;
	if (useKinect)
		inputsSpecified++;
	if (useFile)
		inputsSpecified++;
	if (useDirectory)
		inputsSpecified++;
	if (inputsSpecified != 1) {
		std::cout << "Usage: Please specify a camera, Kinect, file or directory (and only one of them) to run the program. Use -h for help." << std::endl;
		return -1;
	}

	Logger->setVerboseLevelText(verboseLevelText);
	Logger->setVerboseLevelImages(verboseLevelImages);

	auto_ptr<imageio::ImageSource> imageSource;
	if (useCamera)
		imageSource.reset(new imageio::VideoImageSource(deviceId));
	else if (useKinect)
		imageSource.reset(new imageio::KinectImageSource(kinectId));
	else if (useFile)
		imageSource.reset(new imageio::VideoImageSource(filename));
	else if (useDirectory)
		imageSource.reset(new imageio::DirectoryImageSource(directory));

	auto_ptr<FaceTracking> tracker(new FaceTracking(imageSource, svmConfigFile, negativeRtlPatches));
	tracker->run();
	return 0;
}
