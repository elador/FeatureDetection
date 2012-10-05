/*
 * FaceTracking.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "FaceTracking.h"
#include "VideoImageSource.h"
#include "DirectoryImageSource.h"
#include "tracking/ResamplingSampler.h"
#include "tracking/GridSampler.h"
#include "tracking/LowVarianceSampling.h"
#include "tracking/SimpleTransitionModel.h"
#include "tracking/SvmTraining.h"
#include "tracking/FrameBasedSvmTraining.h"
#include "tracking/ApproximateSigmoidParameterComputation.h"
#include "OverlapElimination.h"
#include "tracking/FilteringPositionExtractor.h"
#include "tracking/WeightedMeanPositionExtractor.h"
#include "tracking/Rectangle.h"
#include "tracking/Sample.h"
#include "VDetectorVectorMachine.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "tracking/ChangableDetectorSvm.h"
#include "FdImage.h"
#include "boost/make_shared.hpp"
#include "boost/optional.hpp"
#include <vector>
#include <iostream>
#include <sstream>
#include <cstring>
#ifdef WIN32
	#include "wingettimeofday.h"
#else
	#include <sys/time.h>
#endif

const std::string FaceTracking::svmConfigFile = "/home/poschmann/projects/ffd/config/fdetection/fd_config_fft_fd.mat";
const std::string FaceTracking::negativesFile = "/home/poschmann/projects/ffd/config/nonfaces_1000";
const std::string FaceTracking::videoWindowName = "Image";
const std::string FaceTracking::controlWindowName = "Controls";

FaceTracking::FaceTracking(auto_ptr<ImageSource> imageSource) : imageSource(imageSource) {
	initTracking();
	initGui();
}

FaceTracking::~FaceTracking() {}

void FaceTracking::initTracking() {
	// create SVM training
	svmTraining = boost::make_shared<FrameBasedSvmTraining>(5, 4, negativesFile, 200);

	// create measurement model
	shared_ptr<VDetectorVectorMachine> wvm = make_shared<DetectorWVM>();
	wvm->load(svmConfigFile);
	shared_ptr<VDetectorVectorMachine> svm = make_shared<DetectorSVM>();
	svm->load(svmConfigFile);
	shared_ptr<ChangableDetectorSvm> dynamicSvm = make_shared<ChangableDetectorSvm>();
	dynamicSvm->load(svmConfigFile);
	shared_ptr<OverlapElimination> oe = make_shared<OverlapElimination>();
	oe->load(svmConfigFile);
	measurementModel = make_shared<SelfLearningWvmSvmModel>(wvm, svm, dynamicSvm, oe, svmTraining, 0.85, 0.05);

	// create tracker
	unsigned int count = 800;
	double randomRate = 0.35;
	transitionModel = make_shared<SimpleTransitionModel>(0.2);
	resamplingSampler = make_shared<ResamplingSampler>(count, randomRate, make_shared<LowVarianceSampling>(),
			transitionModel);
	gridSampler = make_shared<GridSampler>(0.2, 0.8, 1.2, 0.1);
	tracker = auto_ptr<CondensationTracker>(new CondensationTracker(resamplingSampler, measurementModel,
			make_shared<FilteringPositionExtractor>(make_shared<WeightedMeanPositionExtractor>())));
}

void FaceTracking::initGui() {
	drawSamples = true;

	cvNamedWindow(controlWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(controlWindowName.c_str(), 750, 50);

	cv::createTrackbar("Self-learning active", controlWindowName, NULL, 1, selfLearningChanged, this);
	cv::setTrackbarPos("Self-learning active", controlWindowName, measurementModel->isSelfLearningActive() ? 1 : 0);

	cv::createTrackbar("Grid/Resampling", controlWindowName, NULL, 1, samplerChanged, this);
	cv::setTrackbarPos("Grid/Resampling", controlWindowName, tracker->getSampler() == gridSampler ? 0 : 1);

	cv::createTrackbar("Sample Count", controlWindowName, NULL, 1000, sampleCountChanged, this);
	cv::setTrackbarPos("Sample Count", controlWindowName, resamplingSampler->getCount());

	cv::createTrackbar("Random Rate", controlWindowName, NULL, 100, randomRateChanged, this);
	cv::setTrackbarPos("Random Rate", controlWindowName, 100 * resamplingSampler->getRandomRate());

	cv::createTrackbar("Scatter * 100", controlWindowName, NULL, 100, scatterChanged, this);
	cv::setTrackbarPos("Scatter * 100", controlWindowName, 100 * transitionModel->getScatter());

	cv::createTrackbar("Draw samples", controlWindowName, NULL, 1, drawSamplesChanged, this);
	cv::setTrackbarPos("Draw samples", controlWindowName, drawSamples ? 1 : 0);
}

void FaceTracking::selfLearningChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->measurementModel->setSelfLearningActive(state == 1);
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
	cv::Scalar& svmIndicatorColor = measurementModel->isUsingDynamicSvm() ? green : red;
	cv::circle(image, cv::Point(10, 10), 5, svmIndicatorColor, -1);
	std::ostringstream patchText;
	patchText << svmTraining->getPositiveSampleCount() << '/' << svmTraining->getRequiredPositiveSampleCount();
	bool enoughSamples = (svmTraining->getPositiveSampleCount() >= svmTraining->getRequiredPositiveSampleCount());
	cv::Scalar& textColor = enoughSamples ? green : red;
	cv::putText(image, patchText.str(), cv::Point(20, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, textColor);
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
			std::cerr << "Could not capture frame" << std::endl;
			stop();
		} else {
			if (first) {
				first = false;
				image.create(frame.rows, frame.cols, frame.type());
			}
			FdImage* myImage = new FdImage();
			myImage->load(&frame);
			gettimeofday(&detStart, 0);
			boost::optional<tracking::Rectangle> face = tracker->process(myImage);
			gettimeofday(&detEnd, 0);
			delete myImage;
			image = frame;
			drawDebug(image);
			cv::Scalar& color = measurementModel->isUsingDynamicSvm() ? green : red;
			if (face)
				cv::rectangle(image, cv::Point(face->getX(), face->getY()),
						cv::Point(face->getX() + face->getWidth(), face->getY() + face->getHeight()), color);
			imshow(videoWindowName, image);
			#ifdef WIN32
				Sleep(10);
			#else
				usleep(10000);
			#endif

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

			int delay = paused ? 0 : 10;
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
	if (argc < 3) {
		std::cout << "Usage: -c device OR -v filename OR -i directory" << std::endl;
		return -1;
	}
	auto_ptr<FaceTracking> tracker;
	if (strcmp("-c", argv[1]) == 0) {
		std::istringstream iss(argv[2]);
		int device;
		iss >> device;
		auto_ptr<ImageSource> imageSource(new VideoImageSource(device));
		tracker.reset(new FaceTracking(imageSource));
	} else if (strcmp("-v", argv[1]) == 0) {
		auto_ptr<ImageSource> imageSource(new VideoImageSource(argv[2]));
		tracker.reset(new FaceTracking(imageSource));
	} else if (strcmp("-i", argv[1]) == 0) {
		auto_ptr<ImageSource> imageSource(new DirectoryImageSource(argv[2]));
		tracker.reset(new FaceTracking(imageSource));
	}
	tracker->run();
	return 0;
}
