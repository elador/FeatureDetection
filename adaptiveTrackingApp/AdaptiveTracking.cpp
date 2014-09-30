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
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/VideoImageSource.hpp"
#include "imageio/CameraImageSource.hpp"
#include "imageio/EmptyAnnotationSource.hpp"
#include "imageio/BobotAnnotationSource.hpp"
#include "imageio/SimpleAnnotationSource.hpp"
#include "imageio/VideoImageSink.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/AgeBasedExampleManagement.hpp"
#include "classification/ConfidenceBasedExampleManagement.hpp"
#include "classification/UnlimitedExampleManagement.hpp"
#include "classification/FixedTrainableProbabilisticSvmClassifier.hpp"
#include "libsvm/LibSvmClassifier.hpp"
#include "condensation/LowVarianceSampling.hpp"
#include "condensation/ExtendedHogBasedMeasurementModel.hpp"
#include "condensation/FilteringStateExtractor.hpp"
#include "condensation/WeightedMeanStateExtractor.hpp"
#include "condensation/Sample.hpp"
#include "boost/program_options.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/optional.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <sstream>
#include <algorithm>

using namespace logging;
using namespace classification;
using namespace std::chrono;
using libsvm::LibSvmClassifier;
using cv::Point;
using cv::Rect;
using cv::Rect_;
using boost::property_tree::info_parser::read_info;
using std::milli;
using std::move;
using std::ostringstream;
using std::istringstream;
using std::invalid_argument;

namespace po = boost::program_options;

const string AdaptiveTracking::videoWindowName = "Image";
const string AdaptiveTracking::controlWindowName = "Controls";

AdaptiveTracking::AdaptiveTracking(unique_ptr<AnnotatedImageSource> imageSource, unique_ptr<ImageSink> imageSink, ptree& config) :
		imageSource(move(imageSource)), imageSink(move(imageSink)) {
	initTracking(config);
	initGui();
}

unique_ptr<ExampleManagement> AdaptiveTracking::createExampleManagement(ptree& config, shared_ptr<BinaryClassifier> classifier, bool positive) {
	if (config.get_value<string>() == "unlimited") {
		return unique_ptr<ExampleManagement>(new UnlimitedExampleManagement(config.get<size_t>("required")));
	} else if (config.get_value<string>() == "agebased") {
		return unique_ptr<ExampleManagement>(new AgeBasedExampleManagement(config.get<size_t>("capacity"), config.get<size_t>("required")));
	} else if (config.get_value<string>() == "confidencebased") {
		return unique_ptr<ExampleManagement>(new ConfidenceBasedExampleManagement(classifier, positive, config.get<size_t>("capacity"), config.get<size_t>("required")));
	} else {
		throw invalid_argument("AdaptiveTracking: invalid example management type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableProbabilisticSvmClassifier> AdaptiveTracking::createSvm(ptree& config) {
	shared_ptr<LibSvmClassifier> svm = make_shared<LibSvmClassifier>(make_shared<LinearKernel>(), config.get<double>("training.C"));
	svm->setPositiveExampleManagement(
			unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("training.positiveExamples"), svm, true)));
	svm->setNegativeExampleManagement(
			unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("training.negativeExamples"), svm, false)));
	return make_shared<FixedTrainableProbabilisticSvmClassifier>(svm,
			config.get<double>("probabilistic.logisticA"), config.get<double>("probabilistic.logisticB"));
}

void AdaptiveTracking::initTracking(ptree& config) {
	// create measurement model
	shared_ptr<TrainableProbabilisticSvmClassifier> classifier = createSvm(config.get_child("measurement.classifier"));
	shared_ptr<ExtendedHogBasedMeasurementModel> measurementModel = make_shared<ExtendedHogBasedMeasurementModel>(classifier);
	measurementModel->setHogParams(
			config.get<size_t>("measurement.cellSize"),
			config.get<size_t>("measurement.cellCount"),
			config.get<bool>("measurement.signedAndUnsigned"),
			config.get<bool>("measurement.interpolateBins"),
			config.get<bool>("measurement.interpolateCells"),
			config.get<int>("measurement.octaveLayerCount"));
	measurementModel->setRejectionThreshold(
			config.get<double>("measurement.rejectionThreshold"));
	measurementModel->setUseSlidingWindow(
			config.get<bool>("measurement.useSlidingWindow"),
			config.get<bool>("measurement.conservativeReInit"));
	measurementModel->setNegativeExampleParams(
			config.get<size_t>("measurement.negativeExampleCount"),
			config.get<size_t>("measurement.initialNegativeExampleCount"),
			config.get<size_t>("measurement.randomExampleCount"),
			config.get<float>("measurement.negativeScoreThreshold"));
	measurementModel->setOverlapThresholds(
			config.get<double>("measurement.positiveOverlapThreshold"),
			config.get<double>("measurement.negativeOverlapThreshold"));
	ExtendedHogBasedMeasurementModel::Adaptation adaptation;
	if (config.get<string>("measurement.adaptation") == "NONE")
		adaptation = ExtendedHogBasedMeasurementModel::Adaptation::NONE;
	else if (config.get<string>("measurement.adaptation") == "POSITION")
		adaptation = ExtendedHogBasedMeasurementModel::Adaptation::POSITION;
	else if (config.get<string>("measurement.adaptation") == "TRAJECTORY")
		adaptation = ExtendedHogBasedMeasurementModel::Adaptation::TRAJECTORY;
	else if (config.get<string>("measurement.adaptation") == "CORRECTED_TRAJECTORY")
		adaptation = ExtendedHogBasedMeasurementModel::Adaptation::CORRECTED_TRAJECTORY;
	else
		throw invalid_argument("AdaptiveTracking: invalid adaptation type: " + config.get<string>("measurement.adaptation"));
	measurementModel->setAdaptation(adaptation,
			config.get<double>("measurement.adaptationThreshold"),
			config.get<double>("measurement.exclusionThreshold"));

	// create transition model
	shared_ptr<TransitionModel> transitionModel;
	if (config.get<string>("transition") == "constantVelocity") {
		simpleTransitionModel = make_shared<ConstantVelocityModel>(
				config.get<double>("transition.positionDeviation"), config.get<double>("transition.sizeDeviation"));
		transitionModel = simpleTransitionModel;
	} else if (config.get<string>("transition") == "opticalFlow") {
		simpleTransitionModel = make_shared<ConstantVelocityModel>(
				config.get<double>("transition.fallback.positionDeviation"), config.get<double>("transition.fallback.sizeDeviation"));
		opticalFlowTransitionModel = make_shared<OpticalFlowTransitionModel>(
				simpleTransitionModel, config.get<double>("transition.positionDeviation"), config.get<double>("transition.sizeDeviation"));
		transitionModel = opticalFlowTransitionModel;
	} else {
		throw invalid_argument("AdaptiveTracking: invalid transition model type: " + config.get<string>("transition"));
	}

	// create tracker
	shared_ptr<StateExtractor> stateExtractor = make_shared<FilteringStateExtractor>(make_shared<WeightedMeanStateExtractor>());
	tracker = unique_ptr<CondensationTracker>(new CondensationTracker(
			transitionModel, measurementModel, make_shared<LowVarianceSampling>(), stateExtractor,
			config.get<size_t>("resampling.particleCount"), config.get<double>("resampling.randomRate"),
			config.get<double>("resampling.minSize"), config.get<double>("resampling.maxSize")));

	if (config.get<string>("initialization") == "manual") {
		initialization = Initialization::MANUAL;
	} else if (config.get<string>("initialization") == "groundtruth") {
		initialization = Initialization::GROUND_TRUTH;
	} else {
		throw invalid_argument("AdaptiveTracking: invalid initialization type: " + config.get<string>("initialization"));
	}
}

void AdaptiveTracking::initGui() {
	drawSamples = false;
	drawFlow = 0;

	cvNamedWindow(videoWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(videoWindowName.c_str(), 50, 50);

	cvNamedWindow(controlWindowName.c_str(), CV_WINDOW_NORMAL);
	cvMoveWindow(controlWindowName.c_str(), 900, 50);

	if (opticalFlowTransitionModel) {
		cv::createTrackbar("Position Deviation * 10", controlWindowName, NULL, 100, positionDeviationChanged, this);
		cv::setTrackbarPos("Position Deviation * 10", controlWindowName, 100 * opticalFlowTransitionModel->getPositionDeviation());

		cv::createTrackbar("Size Deviation * 100", controlWindowName, NULL, 100, sizeDeviationChanged, this);
		cv::setTrackbarPos("Size Deviation * 100", controlWindowName, 100 * opticalFlowTransitionModel->getSizeDeviation());
	} else {
		cv::createTrackbar("Position Deviation * 10", controlWindowName, NULL, 100, positionDeviationChanged, this);
		cv::setTrackbarPos("Position Deviation * 10", controlWindowName, 100 * simpleTransitionModel->getPositionDeviation());

		cv::createTrackbar("Size Deviation * 100", controlWindowName, NULL, 100, sizeDeviationChanged, this);
		cv::setTrackbarPos("Size Deviation * 100", controlWindowName, 100 * simpleTransitionModel->getSizeDeviation());
	}

	cv::createTrackbar("Sample Count", controlWindowName, NULL, 2000, sampleCountChanged, this);
	cv::setTrackbarPos("Sample Count", controlWindowName, tracker->getCount());

	cv::createTrackbar("Random Rate", controlWindowName, NULL, 100, randomRateChanged, this);
	cv::setTrackbarPos("Random Rate", controlWindowName, 100 * tracker->getRandomRate());

	cv::createTrackbar("Draw samples", controlWindowName, NULL, 1, drawSamplesChanged, this);
	cv::setTrackbarPos("Draw samples", controlWindowName, drawSamples ? 1 : 0);

	if (opticalFlowTransitionModel) {
		cv::createTrackbar("Draw flow", controlWindowName, NULL, 3, drawFlowChanged, this);
		cv::setTrackbarPos("Draw flow", controlWindowName, drawFlow);
	}
}

void AdaptiveTracking::positionDeviationChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	if (tracking->opticalFlowTransitionModel)
		tracking->opticalFlowTransitionModel->setPositionDeviation(0.1 * state);
	else
		tracking->simpleTransitionModel->setPositionDeviation(0.1 * state);
}

void AdaptiveTracking::sizeDeviationChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	if (tracking->opticalFlowTransitionModel)
		tracking->opticalFlowTransitionModel->setSizeDeviation(0.01 * state);
	else
		tracking->simpleTransitionModel->setSizeDeviation(0.01 * state);
}

void AdaptiveTracking::sampleCountChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->tracker->setCount(state);
}

void AdaptiveTracking::randomRateChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->tracker->setRandomRate(0.01 * state);
}

void AdaptiveTracking::drawSamplesChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->drawSamples = (state == 1);
}

void AdaptiveTracking::drawFlowChanged(int state, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	tracking->drawFlow = state;
}

void AdaptiveTracking::drawDebug(Mat& image) {
	cv::Scalar black(0, 0, 0); // blue, green, red
	cv::Scalar red(0, 0, 255); // blue, green, red
	cv::Scalar green(0, 255, 0); // blue, green, red
	if (drawSamples) {
		const std::vector<shared_ptr<Sample>>& samples = tracker->getSamples();
		for (const shared_ptr<Sample>& sample : samples) {
			if (!sample->isTarget())
				cv::circle(image, Point(sample->getX(), sample->getY()), 3, black);
		}
		for (const shared_ptr<Sample>& sample : samples) {
			if (sample->isTarget()) {
				cv::Scalar color(0, sample->getWeight() * 255, sample->getWeight() * 255);
				cv::circle(image, Point(sample->getX(), sample->getY()), 3, color);
			}
		}
	}
	if (drawFlow == 1)
		opticalFlowTransitionModel->drawFlow(image, -drawFlow, green, red);
	else if (drawFlow > 1)
		opticalFlowTransitionModel->drawFlow(image, drawFlow - 1, green, red);
}

void AdaptiveTracking::drawCrosshair(Mat& image) {
	if (currentX >= 0 && currentY >= 0 && currentX < image.cols && currentY < image.rows) {
		cv::Scalar gray(127, 127, 127); // blue, green, red
		cv::Scalar black(0, 0, 0); // blue, green, red
		cv::line(image, Point(currentX, 0), Point(currentX, image.rows), gray);
		cv::line(image, Point(0, currentY), Point(image.cols, currentY), gray);
		cv::line(image, Point(currentX, currentY - 5), Point(currentX, currentY + 5), black);
		cv::line(image, Point(currentX - 5, currentY), Point(currentX + 5, currentY), black);
	}
}

void AdaptiveTracking::drawBox(Mat& image) {
	if (storedX >= 0 && storedY >= 0
			&& currentX >= 0 && currentY >= 0 && currentX < image.cols && currentY < image.rows) {
		cv::Scalar color(204, 102, 0); // blue, green, red
		cv::rectangle(image, Point(storedX, storedY), Point(currentX, currentY), color, 2);
	}
}

void AdaptiveTracking::drawGroundTruth(Mat& image, const Rect& target) {
	cv::rectangle(image, target, cv::Scalar(255, 153, 102), 2);
}

void AdaptiveTracking::drawTarget(Mat& image, optional<Rect> target, bool adapted) {
	const cv::Scalar& color = adapted ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
	if (target)
		cv::rectangle(image, *target, color, 2);
}

void AdaptiveTracking::onMouse(int event, int x, int y, int, void* userdata) {
	AdaptiveTracking *tracking = (AdaptiveTracking*)userdata;
	if (tracking->running && !tracking->adaptiveUsable) {
		if (event == cv::EVENT_MOUSEMOVE) {
			tracking->currentX = x;
			tracking->currentY = y;
			Mat& image = tracking->image;
			tracking->frame.copyTo(image);
			tracking->drawBox(image);
			tracking->drawCrosshair(image);
			imshow(videoWindowName, image);
		} else if (event == cv::EVENT_LBUTTONDOWN) {
			if (tracking->storedX < 0 || tracking->storedY < 0) {
				tracking->storedX = x;
				tracking->storedY = y;
			}
		} else if (event == cv::EVENT_LBUTTONUP) {
			Logger& log = Loggers->getLogger("app");
			Rect position(std::min(tracking->storedX, tracking->currentX), std::min(tracking->storedY, tracking->currentY),
					abs(tracking->currentX - tracking->storedX), abs(tracking->currentY - tracking->storedY));

			if (position.width != 0 && position.height != 0) {
				int tries = 0;
				while (tries < 10 && !tracking->adaptiveUsable) {
					tries++;
					tracking->adaptiveUsable = tracking->tracker->initialize(tracking->frame, position);
				}
				if (tracking->adaptiveUsable) {
					log.info("Initialized adaptive tracking after " + std::to_string(tries) + " tries");
					tracking->storedX = -1;
					tracking->storedY = -1;
					tracking->currentX = -1;
					tracking->currentY = -1;
					Mat& image = tracking->image;
					tracking->frame.copyTo(image);
					tracking->drawTarget(image, optional<Rect>(position), true);
					imshow(videoWindowName, image);
				} else {
					log.warn("Could not initialize tracker after " + std::to_string(tries) + " tries (patch too small/big?)");
					std::cerr << "Could not initialize tracker - press 'q' to quit program" << std::endl;
					tracking->stop();
					while ('q' != (char)cv::waitKey(10));
				}
			}
		}
	}
}

void AdaptiveTracking::run() {
	Logger& log = Loggers->getLogger("app");
	running = true;
	paused = false;
	adaptiveUsable = false;
	double overlapSum = 0;
	size_t hitCount = 0;
	size_t frameCount = 0;

	while (running) {

		// initialization (manual or via ground truth)
		if (initialization == Initialization::MANUAL) {
			storedX = -1;
			storedY = -1;
			currentX = -1;
			currentY = -1;
			cv::setMouseCallback(videoWindowName, onMouse, this);
			while (running && !adaptiveUsable) {
				if (!imageSource->next()) {
					std::cerr << "Could not capture frame - press 'q' to quit program" << std::endl;
					stop();
					while ('q' != (char)cv::waitKey(10));
				} else {
					frame = imageSource->getImage();
					frame.copyTo(image);
					drawGroundTruth(image, imageSource->getAnnotation());
					drawBox(image);
					drawCrosshair(image);
					imshow(videoWindowName, image);
					if (imageSink.get() != 0)
						imageSink->add(image);

					int delay = paused ? 0 : 5;
					char c = (char)cv::waitKey(delay);
					if (c == 'p')
						paused = !paused;
					else if (c == 'q')
						stop();
				}
			}
		} else if (initialization == Initialization::GROUND_TRUTH) {
			int tries = 0;
			int frameIndex = 0;
			while (running && !adaptiveUsable) {
				if (!imageSource->next()) {
					std::cerr << "Could not capture frame - press 'q' to quit program" << std::endl;
					stop();
					while ('q' != (char)cv::waitKey(10));
				} else {
					frame = imageSource->getImage();
					frame.copyTo(image);
					Rect truth = imageSource->getAnnotation();
					drawGroundTruth(image, truth);
					if (truth.area() > 0
							&& truth.x >= 0 && truth.y >= 0
							&& truth.br().x < image.cols && truth.br().y < image.rows) {
						tries++;
						optional<Rect> position = tracker->initialize(frame, truth);
						adaptiveUsable = position;
						drawTarget(image, position, true);
						if (adaptiveUsable) {
							log.info("Initialized adaptive tracking after " + std::to_string(tries) + " tries");
							frameCount++;
							double intersectionArea = (truth & *position).area();
							double unionArea = truth.area() + position->area() - intersectionArea;
							double overlap = unionArea > 0 ? intersectionArea / unionArea : 0;
							overlapSum += overlap;
							if (overlap >= 0.5)
								hitCount++;
						} else if (tries == 10) {
							log.warn("Could not initialize tracker after " + std::to_string(tries) + " tries (patch too small/big?)");
							std::cerr << "Could not initialize tracker - press 'q' to quit program" << std::endl;
							stop();
							while ('q' != (char)cv::waitKey(10));
						}
					}
					imshow(videoWindowName, image);
					if (imageSink.get() != 0)
						imageSink->add(image);

					int delay = paused ? 0 : 5;
					char c = (char)cv::waitKey(delay);
					if (c == 'p')
						paused = !paused;
					else if (c == 'q')
						stop();
					frameIndex++;
				}
			}
		}

		duration<double> allIterationTime;
		duration<double> allCondensationTime;
		int frames = 0;

		// adaptive tracking
		while (running) {
			steady_clock::time_point frameStart = steady_clock::now();

			if (!imageSource->next()) {
				std::cerr << "Could not capture frame - press 'q' to quit program" << std::endl;
				stop();
				while ('q' != (char)cv::waitKey(10));
			} else {
				frames++;
				frame = imageSource->getImage();
				Rect truth = imageSource->getAnnotation();
				steady_clock::time_point condensationStart = steady_clock::now();
				bool adapted = false;
				optional<Rect> position;
				position = tracker->process(frame);
				adapted = tracker->hasAdapted();
				steady_clock::time_point condensationEnd = steady_clock::now();

				frameCount++;
				if (position) {
					double intersectionArea = (truth & *position).area();
					double unionArea = truth.area() + position->area() - intersectionArea;
					double overlap = unionArea > 0 ? intersectionArea / unionArea : 0;
					overlapSum += overlap;
					if (overlap >= 0.5)
						hitCount++;
				}

				frame.copyTo(image);
				drawDebug(image);
				drawGroundTruth(image, truth);
				drawTarget(image, position, adapted);
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
				else if (c == 'r') {
					tracker->reset();
					adaptiveUsable = false;
					break;
				}
			}
		}
		std::cout << "hit rate: " << (100 * static_cast<double>(hitCount) / frameCount) << "%" << std::endl;
		std::cout << "average overlap: " << (100 * overlapSum / frameCount) << "%" << std::endl;
	}
}

void AdaptiveTracking::stop() {
	running = false;
}

int main(int argc, char *argv[]) {
	int verboseLevelText;
	int verboseLevelImages;
	int deviceId;
	string filename, directory, groundTruthFilename;
	bool useCamera = false, useFile = false, useDirectory = false, useGroundTruth = false, bobot = false;
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
			("ground-truth,g", po::value<string>(&groundTruthFilename), "Name of a file containing ground truth information in BoBoT format")
			("bobot,b", "Flag for indicating BoBoT format on the ground truth file")
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
		if (vm.count("ground-truth"))
			useGroundTruth = true;
		if (vm.count("bobot"))
			bobot = true;
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return -1;
	}

	int inputsSpecified = 0;
	if (useCamera)
		inputsSpecified++;
	if (useFile)
		inputsSpecified++;
	if (useDirectory)
		inputsSpecified++;
	if (inputsSpecified != 1) {
		std::cout << "Usage: Please specify a camera, file or directory (and only one of them) to run the program. Use -h for help." << std::endl;
		return -1;
	}

	Loggers->getLogger("app").addAppender(make_shared<ConsoleAppender>(LogLevel::Info));

	shared_ptr<ImageSource> imageSource;
	if (useCamera)
		imageSource.reset(new CameraImageSource(deviceId));
	else if (useFile)
		imageSource.reset(new VideoImageSource(filename));
	else if (useDirectory)
		imageSource.reset(new DirectoryImageSource(directory));
	shared_ptr<AnnotationSource> annotationSource;
	if (useGroundTruth && bobot)
		annotationSource.reset(new BobotAnnotationSource(groundTruthFilename, imageSource));
	else if (useGroundTruth)
		annotationSource.reset(new SimpleAnnotationSource(groundTruthFilename));
	else
		annotationSource.reset(new EmptyAnnotationSource());
	unique_ptr<AnnotatedImageSource> annotatedImageSource(new AnnotatedImageSource(imageSource, annotationSource));

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
	config.put("tracking.initialization", useGroundTruth ? "groundtruth" : "manual");
	try {
		unique_ptr<AdaptiveTracking> tracker(new AdaptiveTracking(move(annotatedImageSource), move(imageSink), config.get_child("tracking")));
		tracker->run();
	} catch (std::exception& exc) {
		Loggers->getLogger("app").error(string("A wild exception appeared: ") + exc.what());
		throw;
	}
	return 0;
}
