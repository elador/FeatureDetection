/*
 * AdaptiveTracking.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "AdaptiveTracking.hpp"
#include "logging/LoggerFactory.hpp"
#include "logging/Logger.hpp"
#include "logging/ConsoleAppender.hpp"
#include "imageio/VideoImageSource.hpp"
#include "imageio/KinectImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/VideoImageSink.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/FilteringFeatureExtractor.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/HistEq64Filter.hpp"
#include "imageprocessing/WhiteningFilter.hpp"
#include "imageprocessing/ZeroMeanUnitVarianceFilter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"
#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/RbfKernel.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/FrameBasedTrainableSvmClassifier.hpp"
#include "classification/TrainableProbabilisticTwoStageClassifier.hpp"
#include "classification/FixedSizeTrainableSvmClassifier.hpp"
#include "classification/FixedTrainableProbabilisticSvmClassifier.hpp"
#include "condensation/ResamplingSampler.hpp"
#include "condensation/GridSampler.hpp"
#include "condensation/LowVarianceSampling.hpp"
#include "condensation/SimpleTransitionModel.hpp"
#include "condensation/WvmSvmModel.hpp"
#include "condensation/SelfLearningMeasurementModel.hpp"
#include "condensation/PositionDependentMeasurementModel.hpp"
#include "condensation/FilteringPositionExtractor.hpp"
#include "condensation/WeightedMeanPositionExtractor.hpp"
#include "condensation/Sample.hpp"
#include "boost/program_options.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/optional.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <sstream>

using namespace logging;
using namespace imageprocessing;
using namespace classification;
using namespace std::chrono;
using boost::property_tree::info_parser::read_info;
using std::milli;
using std::move;
using std::ostringstream;

namespace po = boost::program_options;

const string AdaptiveTracking::videoWindowName = "Image";
const string AdaptiveTracking::controlWindowName = "Controls";

AdaptiveTracking::AdaptiveTracking(unique_ptr<ImageSource> imageSource, unique_ptr<ImageSink> imageSink, ptree config) :
		imageSource(move(imageSource)), imageSink(move(imageSink)) {
	initTracking(config);
	initGui();
}

AdaptiveTracking::~AdaptiveTracking() {}

shared_ptr<Kernel> AdaptiveTracking::createKernel(ptree config) {
	if (config.get_value<string>() == "rbf") {
		return make_shared<RbfKernel>(config.get<double>("gamma"));
	} else if (config.get_value<string>() == "poly") {
		return make_shared<PolynomialKernel>(
				config.get<double>("alpha"), config.get<double>("constant"), config.get<double>("degree"));
	} else {
		throw invalid_argument("AdaptiveTracking: invalid kernel type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableSvmClassifier> AdaptiveTracking::createTrainableSvm(shared_ptr<Kernel> kernel, ptree config) {
	shared_ptr<TrainableSvmClassifier> trainableSvm;
	if (config.get_value<string>() == "fixedsize") {
		trainableSvm = make_shared<FixedSizeTrainableSvmClassifier>(kernel, config.get<double>("constraintsViolationCosts"),
				config.get<unsigned int>("positiveExamples"), config.get<unsigned int>("negativeExamples"), config.get<unsigned int>("minPositiveExamples"));
	} else if (config.get_value<string>() == "framebased") {
		trainableSvm = make_shared<FrameBasedTrainableSvmClassifier>(kernel, config.get<double>("constraintsViolationCosts"),
				config.get<int>("frameLength"), config.get<float>("minAvgSamples"));
	} else {
		throw invalid_argument("AdaptiveTracking: invalid training type: " + config.get_value<string>());
	}
	optional<ptree&> negativesConfig = config.get_child_optional("staticNegatives");
	if (negativesConfig && negativesConfig->get_value<bool>()) {
		trainableSvm->loadStaticNegatives(negativesConfig->get<string>("filename"),
				negativesConfig->get<int>("amount"), negativesConfig->get<double>("scale"));
	}
	return trainableSvm;
}

shared_ptr<TrainableProbabilisticClassifier> AdaptiveTracking::createClassifier(
		shared_ptr<TrainableSvmClassifier> trainableSvm, ptree config) {
	if (config.get_value<string>() == "default")
		return make_shared<TrainableProbabilisticSvmClassifier>(trainableSvm);
	else if (config.get_value<string>() == "fixed")
		return make_shared<FixedTrainableProbabilisticSvmClassifier>(trainableSvm);
	else
		throw invalid_argument("AdaptiveTracking: invalid classifier type: " + config.get_value<string>());
}

void AdaptiveTracking::initTracking(ptree config) {
	// create feature extractors
	shared_ptr<DirectPyramidFeatureExtractor> patchExtractor = make_shared<DirectPyramidFeatureExtractor>(
			config.get<int>("pyramid.patch.width"), config.get<int>("pyramid.patch.height"),
			config.get<int>("pyramid.patch.minWidth"), config.get<int>("pyramid.patch.maxWidth"),
			config.get<double>("pyramid.scaleFactor"));
	patchExtractor->addImageFilter(make_shared<GrayscaleFilter>());

	shared_ptr<FilteringFeatureExtractor> staticFeatureExtractor = make_shared<FilteringFeatureExtractor>(patchExtractor);
	staticFeatureExtractor->addPatchFilter(make_shared<HistEq64Filter>());

	shared_ptr<FilteringFeatureExtractor> adaptiveFeatureExtractor = make_shared<FilteringFeatureExtractor>(patchExtractor);
	if (config.get<string>("feature.adaptive") == "histeq") {
		adaptiveFeatureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
	} else if (config.get<string>("feature.adaptive") == "whi") {
		adaptiveFeatureExtractor->addPatchFilter(make_shared<WhiteningFilter>());
		adaptiveFeatureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
		adaptiveFeatureExtractor->addPatchFilter(make_shared<ZeroMeanUnitVarianceFilter>());
	} else {
		throw invalid_argument("AdaptiveTracking: invalid feature type: " + config.get<string>("feature.adaptive"));
	}

	// create static measurement model
	string classifierFile = config.get<string>("measurement.static.classifier.configFile");
	string thresholdsFile = config.get<string>("measurement.static.classifier.thresholdsFile");
	shared_ptr<ProbabilisticWvmClassifier> wvm = ProbabilisticWvmClassifier::loadMatlab(classifierFile, thresholdsFile);
	shared_ptr<ProbabilisticSvmClassifier> svm = ProbabilisticSvmClassifier::loadMatlab(classifierFile, thresholdsFile);
	staticMeasurementModel = make_shared<WvmSvmModel>(staticFeatureExtractor, wvm, svm);

	// create adaptive measurement model
	shared_ptr<Kernel> kernel = createKernel(config.get_child("measurement.adaptive.classifier.kernel"));
	shared_ptr<TrainableSvmClassifier> trainableSvm = createTrainableSvm(kernel, config.get_child("measurement.adaptive.classifier.training"));
	shared_ptr<TrainableProbabilisticClassifier> classifier = createClassifier(trainableSvm, config.get_child("measurement.adaptive.classifier.probabilistic"));
	optional<bool> withWvm = config.get_optional<bool>("measurement.adaptive.classifier.withWvm");
	if (withWvm && *withWvm)
		classifier = make_shared<TrainableProbabilisticTwoStageClassifier>(wvm, classifier);
	if (config.get<string>("measurement.adaptive") == "selfLearning") {
		adaptiveMeasurementModel = make_shared<SelfLearningMeasurementModel>(adaptiveFeatureExtractor, classifier,
				config.get<double>("measurement.adaptive.positiveThreshold"), config.get<double>("measurement.adaptive.negativeThreshold"));
	} else if (config.get<string>("measurement.adaptive") == "positionDependent") {
		adaptiveMeasurementModel = make_shared<PositionDependentMeasurementModel>(adaptiveFeatureExtractor, classifier,
				config.get<int>("measurement.adaptive.startFrameCount"), config.get<int>("measurement.adaptive.stopFrameCount"),
				config.get<float>("measurement.adaptive.positiveOffsetFactor"), config.get<float>("measurement.adaptive.negativeOffsetFactor"),
				config.get<bool>("measurement.adaptive.sampleNegativesAroundTarget"), config.get<bool>("measurement.adaptive.sampleFalsePositives"),
				config.get<unsigned int>("measurement.adaptive.randomNegatives"));
	} else {
		throw invalid_argument("AdaptiveTracking: invalid adaptive measurement model type: " + config.get<string>("measurement.adaptive"));
	}

	// create tracker
	transitionModel = make_shared<SimpleTransitionModel>(config.get<double>("transition.scatter"));
	resamplingSampler = make_shared<ResamplingSampler>(
			config.get<unsigned int>("resampling.particleCount"), config.get<double>("resampling.randomRate"), make_shared<LowVarianceSampling>(),
			transitionModel, config.get<double>("resampling.minSize"), config.get<double>("resampling.maxSize"));
	gridSampler = make_shared<GridSampler>(config.get<int>("pyramid.patch.minWidth"), config.get<int>("pyramid.patch.maxWidth"),
			1 / config.get<double>("pyramid.scaleFactor"), 0.1);
	tracker = unique_ptr<PartiallyAdaptiveCondensationTracker>(new PartiallyAdaptiveCondensationTracker(
			resamplingSampler, staticMeasurementModel, adaptiveMeasurementModel,
			make_shared<FilteringPositionExtractor>(make_shared<WeightedMeanPositionExtractor>())));
//	tracker->setUseAdaptiveModel(false);
}

void AdaptiveTracking::initGui() {
	drawSamples = true;

	cvNamedWindow(videoWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(videoWindowName.c_str(), 50, 50);

	cvNamedWindow(controlWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(controlWindowName.c_str(), 900, 50);

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

void AdaptiveTracking::adaptiveChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->tracker->setUseAdaptiveModel(state == 1);
}

void AdaptiveTracking::samplerChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	if (state == 0)
		tracking->tracker->setSampler(tracking->gridSampler);
	else
		tracking->tracker->setSampler(tracking->resamplingSampler);
}

void AdaptiveTracking::sampleCountChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->resamplingSampler->setCount(state);
}

void AdaptiveTracking::randomRateChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->resamplingSampler->setRandomRate(0.01 * state);
}

void AdaptiveTracking::scatterChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->transitionModel->setScatter(0.01 * state);
}

void AdaptiveTracking::drawSamplesChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->drawSamples = (state == 1);
}

void AdaptiveTracking::drawDebug(cv::Mat& image) {
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

void AdaptiveTracking::run() {
	Logger& log = Loggers->getLogger("app");
	running = true;
	paused = false;

	bool first = true;
	cv::Mat frame, image;
	cv::Scalar green(0, 255, 0); // blue, green, red
	cv::Scalar red(0, 0, 255); // blue, green, red

	duration<double> allIterationTime;
	duration<double> allCondensationTime;
	int frames = 0;
	std::cout.precision(2);

	while (running) {
		frames++;
		steady_clock::time_point frameStart = steady_clock::now();
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
			steady_clock::time_point condensationStart = steady_clock::now();
			boost::optional<Rect> face = tracker->process(frame);
			steady_clock::time_point condensationEnd = steady_clock::now();
			image = frame;
			drawDebug(image);
			cv::Scalar& color = tracker->wasUsingAdaptiveModel() ? green : red;
			if (face)
				cv::rectangle(image, *face, color);
			imshow(videoWindowName, image);
			if (imageSink.get() != 0)
				imageSink->add(image);
			steady_clock::time_point frameEnd = steady_clock::now();

			milliseconds iterationTime = duration_cast<milliseconds>(frameEnd - frameStart);
			milliseconds condensationTime = duration_cast<milliseconds>(condensationEnd - condensationStart);
			allIterationTime += iterationTime;
			allCondensationTime += condensationTime;
			float iterationFps = frames / allIterationTime.count();
			float condensationFps = frames / allCondensationTime.count();

			ostringstream text;
			text.precision(2);
			text << frames << " frame: " << iterationTime.count() << " ms (" << iterationFps << " fps);"
					<< " condensation: " << condensationTime.count() << " ms (" << condensationFps << " fps)";
			log.info(text.str());

			int delay = paused ? 0 : 5;
			char c = (char)cv::waitKey(delay);
			if (c == 'p')
				paused = !paused;
			else if (c == 'q')
				stop();
		}
	}
}

void AdaptiveTracking::stop() {
	running = false;
}

int main(int argc, char *argv[]) {
	int verboseLevelText;
	int verboseLevelImages;
	int deviceId, kinectId;
	string filename, directory;
	bool useCamera = false, useKinect = false, useFile = false, useDirectory = false;
	string configFile;
	string outputFile;
	int outputFps = -1;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "Produce help message")
			("verbose-text,v", po::value<int>(&verboseLevelText)->implicit_value(2)->default_value(0,"minimal text output"), "Enable text-verbosity (optionally specify level)")
			("verbose-images,w", po::value<int>(&verboseLevelImages)->implicit_value(2)->default_value(0,"minimal image output"), "Enable image-verbosity (optionally specify level)")
			("filename,f", po::value< string >(&filename), "A filename of a video to run the tracking")
			("directory,i", po::value< string >(&directory), "Use a directory as input")
			("device,d", po::value<int>(&deviceId)->implicit_value(0), "A camera device ID for use with the OpenCV camera driver")
			("kinect,k", po::value<int>(&kinectId)->implicit_value(0), "Windows only: Use a Kinect as camera. Optionally specify a device ID.")
			("config,c", po::value< string >(&configFile)->default_value("default.cfg","default.cfg"), "The filename to the config file.")
			("output,o", po::value< string >(&outputFile)->default_value("","none"), "Filename to a video file for storing the image data.")
			("output-fps,r", po::value<int>(&outputFps)->default_value(-1), "The framerate of the output video.")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
			std::cout << "Usage: adaptiveTrackingApp [options]" << std::endl;
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

	Loggers->getLogger("app").addAppender(make_shared<ConsoleAppender>(loglevel::INFO));

	unique_ptr<ImageSource> imageSource;
	if (useCamera)
		imageSource.reset(new VideoImageSource(deviceId));
	else if (useKinect)
		imageSource.reset(new KinectImageSource(kinectId));
	else if (useFile)
		imageSource.reset(new VideoImageSource(filename));
	else if (useDirectory)
		imageSource.reset(new DirectoryImageSource(directory));

	unique_ptr<ImageSink> imageSink;
	if (outputFile != "") {
		if (outputFps < 0) {
			std::cout << "Usage: You have to specify the framerate of the output video file by using option -r. Use -h for help." << std::endl;
			return -1;
		}
		imageSink.reset(new VideoImageSink(outputFile, outputFps));
	}

	ptree config;
	read_info(configFile, config);
	unique_ptr<AdaptiveTracking> tracker(new AdaptiveTracking(move(imageSource), move(imageSink), config.get_child("tracking")));
	tracker->run();
	return 0;
}
