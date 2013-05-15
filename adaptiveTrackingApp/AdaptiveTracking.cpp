/*
 * AdaptiveTracking.cpp
 *
 *  Created on: 14.05.2013
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
	string classifierFile = config.get<string>("initial.measurement.classifier.configFile");
	string thresholdsFile = config.get<string>("initial.measurement.classifier.thresholdsFile");
	shared_ptr<ProbabilisticWvmClassifier> wvm = ProbabilisticWvmClassifier::loadMatlab(classifierFile, thresholdsFile);
	shared_ptr<ProbabilisticSvmClassifier> svm = ProbabilisticSvmClassifier::loadMatlab(classifierFile, thresholdsFile);
	staticMeasurementModel = make_shared<WvmSvmModel>(staticFeatureExtractor, wvm, svm);

	// create adaptive measurement model
	shared_ptr<Kernel> kernel = createKernel(config.get_child("adaptive.measurement.classifier.kernel"));
	shared_ptr<TrainableSvmClassifier> trainableSvm = createTrainableSvm(kernel, config.get_child("adaptive.measurement.classifier.training"));
	shared_ptr<TrainableProbabilisticClassifier> classifier = createClassifier(trainableSvm, config.get_child("adaptive.measurement.classifier.probabilistic"));
	optional<bool> withWvm = config.get_optional<bool>("adaptive.measurement.classifier.withWvm");
	if (withWvm && *withWvm)
		classifier = make_shared<TrainableProbabilisticTwoStageClassifier>(wvm, classifier);
	if (config.get<string>("adaptive.measurement") == "selfLearning") {
		adaptiveMeasurementModel = make_shared<SelfLearningMeasurementModel>(adaptiveFeatureExtractor, classifier,
				config.get<double>("adaptive.measurement.positiveThreshold"), config.get<double>("adaptive.measurement.negativeThreshold"));
	} else if (config.get<string>("adaptive.measurement") == "positionDependent") {
		adaptiveMeasurementModel = make_shared<PositionDependentMeasurementModel>(adaptiveFeatureExtractor, classifier,
				config.get<int>("adaptive.measurement.startFrameCount"), config.get<int>("adaptive.measurement.stopFrameCount"),
				config.get<float>("adaptive.measurement.positiveOffsetFactor"), config.get<float>("adaptive.measurement.negativeOffsetFactor"),
				config.get<bool>("adaptive.measurement.sampleNegativesAroundTarget"), config.get<bool>("adaptive.measurement.sampleFalsePositives"),
				config.get<unsigned int>("adaptive.measurement.randomNegatives"));
	} else {
		throw invalid_argument("AdaptiveTracking: invalid adaptive measurement model type: " + config.get<string>("adaptive.measurement"));
	}

	// create tracker
	transitionModel = make_shared<SimpleTransitionModel>(
			config.get<double>("transition.positionScatter"), config.get<double>("transition.velocityScatter"));
	initialResamplingSampler = make_shared<ResamplingSampler>(
			config.get<unsigned int>("initial.resampling.particleCount"), config.get<double>("initial.resampling.randomRate"),
			make_shared<LowVarianceSampling>(), transitionModel,
			config.get<double>("initial.resampling.minSize"), config.get<double>("initial.resampling.maxSize"));
	adaptiveResamplingSampler = make_shared<ResamplingSampler>(
			config.get<unsigned int>("adaptive.resampling.particleCount"), config.get<double>("adaptive.resampling.randomRate"),
			make_shared<LowVarianceSampling>(), transitionModel,
			config.get<double>("adaptive.resampling.minSize"), config.get<double>("adaptive.resampling.maxSize"));
	gridSampler = make_shared<GridSampler>(config.get<int>("pyramid.patch.minWidth"), config.get<int>("pyramid.patch.maxWidth"),
			1 / config.get<double>("pyramid.scaleFactor"), 0.1);
	shared_ptr<PositionExtractor> positionExtractor
			= make_shared<FilteringPositionExtractor>(make_shared<WeightedMeanPositionExtractor>());
	initialTracker = unique_ptr<CondensationTracker>(new CondensationTracker(
			initialResamplingSampler, staticMeasurementModel, positionExtractor));
	adaptiveTracker = unique_ptr<AdaptiveCondensationTracker>(new AdaptiveCondensationTracker(
			adaptiveResamplingSampler, adaptiveMeasurementModel, positionExtractor,
			config.get<unsigned int>("adaptive.resampling.particleCount")));
	useAdaptive = true;
}

void AdaptiveTracking::initGui() {
	drawSamples = true;

	cvNamedWindow(videoWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(videoWindowName.c_str(), 50, 50);

	cvNamedWindow(controlWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(controlWindowName.c_str(), 900, 50);

	cv::createTrackbar("Adaptive", controlWindowName, NULL, 1, adaptiveChanged, this);
	cv::setTrackbarPos("Adaptive", controlWindowName, useAdaptive ? 1 : 0);

	cv::createTrackbar("Position Scatter * 100", controlWindowName, NULL, 100, positionScatterChanged, this);
	cv::setTrackbarPos("Position Scatter * 100", controlWindowName, 100 * transitionModel->getPositionScatter());

	cv::createTrackbar("Velocity Scatter * 100", controlWindowName, NULL, 100, velocityScatterChanged, this);
	cv::setTrackbarPos("Velocity Scatter * 100", controlWindowName, 100 * transitionModel->getVelocityScatter());

	cv::createTrackbar("Initial Grid/Resampling", controlWindowName, NULL, 1, initialSamplerChanged, this);
	cv::setTrackbarPos("Initial Grid/Resampling", controlWindowName, initialTracker->getSampler() == gridSampler ? 0 : 1);

	cv::createTrackbar("Initial Sample Count", controlWindowName, NULL, 2000, initialSampleCountChanged, this);
	cv::setTrackbarPos("Initial Sample Count", controlWindowName, initialResamplingSampler->getCount());

	cv::createTrackbar("Initial Random Rate", controlWindowName, NULL, 100, initialRandomRateChanged, this);
	cv::setTrackbarPos("Initial Random Rate", controlWindowName, 100 * initialResamplingSampler->getRandomRate());

	cv::createTrackbar("Adaptive Grid/Resampling", controlWindowName, NULL, 1, adaptiveSamplerChanged, this);
	cv::setTrackbarPos("Adaptive Grid/Resampling", controlWindowName, adaptiveTracker->getSampler() == gridSampler ? 0 : 1);

	cv::createTrackbar("Adaptive Sample Count", controlWindowName, NULL, 2000, adaptiveSampleCountChanged, this);
	cv::setTrackbarPos("Adaptive Sample Count", controlWindowName, adaptiveResamplingSampler->getCount());

	cv::createTrackbar("Adaptive Random Rate", controlWindowName, NULL, 100, adaptiveRandomRateChanged, this);
	cv::setTrackbarPos("Adaptive Random Rate", controlWindowName, 100 * adaptiveResamplingSampler->getRandomRate());

	cv::createTrackbar("Draw samples", controlWindowName, NULL, 1, drawSamplesChanged, this);
	cv::setTrackbarPos("Draw samples", controlWindowName, drawSamples ? 1 : 0);
}

void AdaptiveTracking::adaptiveChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->useAdaptive = (state == 1);
}

void AdaptiveTracking::positionScatterChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->transitionModel->setPositionScatter(0.01 * state);
}

void AdaptiveTracking::velocityScatterChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->transitionModel->setVelocityScatter(0.01 * state);
}

void AdaptiveTracking::initialSamplerChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	if (state == 0)
		tracking->initialTracker->setSampler(tracking->gridSampler);
	else
		tracking->initialTracker->setSampler(tracking->initialResamplingSampler);
}

void AdaptiveTracking::initialSampleCountChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->initialResamplingSampler->setCount(state);
}

void AdaptiveTracking::initialRandomRateChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->initialResamplingSampler->setRandomRate(0.01 * state);
}

void AdaptiveTracking::adaptiveSamplerChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	if (state == 0)
		tracking->adaptiveTracker->setSampler(tracking->gridSampler);
	else
		tracking->adaptiveTracker->setSampler(tracking->adaptiveResamplingSampler);
}

void AdaptiveTracking::adaptiveSampleCountChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->adaptiveResamplingSampler->setCount(state);
}

void AdaptiveTracking::adaptiveRandomRateChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->adaptiveResamplingSampler->setRandomRate(0.01 * state);
}

void AdaptiveTracking::drawSamplesChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->drawSamples = (state == 1);
}

void AdaptiveTracking::drawDebug(cv::Mat& image, bool usedAdaptive) {
	cv::Scalar black(0, 0, 0); // blue, green, red
	cv::Scalar red(0, 0, 255); // blue, green, red
	cv::Scalar green(0, 255, 0); // blue, green, red
	if (drawSamples) {
		const std::vector<Sample>& samples = usedAdaptive ? adaptiveTracker->getSamples() : initialTracker->getSamples();
		for (auto sit = samples.cbegin(); sit < samples.cend(); ++sit) {
			if (!sit->isObject())
				cv::circle(image, cv::Point(sit->getX(), sit->getY()), 3, black);
//				cv::line(image, cv::Point(sit->getX() - sit->getVx(), sit->getY() - sit->getVy()), cv::Point(sit->getX(), sit->getY()), black);
		}
		for (auto sit = samples.cbegin(); sit < samples.cend(); ++sit) {
			if (sit->isObject()) {
				cv::Scalar color(0, sit->getWeight() * 255, sit->getWeight() * 255);
				cv::circle(image, cv::Point(sit->getX(), sit->getY()), 3, color);
//				cv::line(image, cv::Point(sit->getX() - sit->getVx(), sit->getY() - sit->getVy()), cv::Point(sit->getX(), sit->getY()), color);
			}
		}
	}
}

void AdaptiveTracking::run() {
	Logger& log = Loggers->getLogger("app");
	running = true;
	paused = false;
	adaptiveUsable = false;

	bool first = true;
	cv::Mat frame, image;
	cv::Scalar green(0, 255, 0); // blue, green, red
	cv::Scalar red(0, 0, 255); // blue, green, red

	duration<double> allIterationTime;
	duration<double> allCondensationTime;
	int frames = 0;

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
			bool usedAdaptive = false;
			boost::optional<Rect> face;
			boost::optional<Sample> state;
			if (useAdaptive) {
				if (adaptiveUsable) {
					face = adaptiveTracker->process(frame);
					state = adaptiveTracker->getState();
					usedAdaptive = true;
				} else {
					face = initialTracker->process(frame);
					state = initialTracker->getState();
					if (face)
						adaptiveUsable = adaptiveTracker->initialize(frame, *face);
				}
			} else {
				face = initialTracker->process(frame);
				state = initialTracker->getState();
			}
			steady_clock::time_point condensationEnd = steady_clock::now();
			image = frame;
			drawDebug(image, usedAdaptive);
			cv::Scalar& color = usedAdaptive ? green : red;
			if (face) {
				cv::rectangle(image, *face, color);
				cv::line(image, cv::Point(state->getX() - state->getVx(), state->getY() - state->getVy()), cv::Point(state->getX(), state->getY()), color);
			}
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
