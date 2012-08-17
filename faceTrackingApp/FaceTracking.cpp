/*
 * FaceTracking.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "FaceTracking.h"
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
#include <sys/time.h>

const std::string FaceTracking::svmConfigFile = "/home/poschmann/projects/ffd/config/fdetection/fd_config_fft_fd.mat";
const std::string FaceTracking::negativesFile = "/home/poschmann/projects/ffd/config/nonfaces_1000";
const std::string FaceTracking::videoWindowName = "Image";
const std::string FaceTracking::controlWindowName = "Controls";

FaceTracking::FaceTracking(int device) {
	init("", device, true, false);
	initGui();
}

FaceTracking::FaceTracking(std::string video, bool realtime) {
	init(video, 0, false, realtime);
	initGui();
}

FaceTracking::~FaceTracking() {}

void FaceTracking::init(std::string video, int device, bool cam, bool realtime) {
	this->video = video;
	this->device = device;
	this->cam = cam;
	this->realtime = realtime;

	frameWidth = 640;
	frameHeight = 480;

	// create SVM training
	svmTraining = make_shared<FrameBasedSvmTraining>(5, 4, negativesFile, 200);

	// create measurement model
	shared_ptr<VDetectorVectorMachine> wvm = make_shared<DetectorWVM>();
	wvm->load(svmConfigFile);
	shared_ptr<VDetectorVectorMachine> svm = make_shared<DetectorSVM>();
	svm->load(svmConfigFile);
	shared_ptr<ChangableDetectorSvm> dynamicSvm = make_shared<ChangableDetectorSvm>();
	dynamicSvm->load(svmConfigFile);
	shared_ptr<OverlapElimination> oe = make_shared<OverlapElimination>();
	oe->load(svmConfigFile);
	measurementModel = make_shared<SelfLearningWvmOeSvmModel>(wvm, svm, dynamicSvm, oe, svmTraining);

	// create tracker
	unsigned int count = 800;
	double randomRate = 0.3;
	resamplingSampler = make_shared<ResamplingSampler>(count, randomRate, make_shared<LowVarianceSampling>(),
			make_shared<SimpleTransitionModel>(0.2));
	gridSampler = make_shared<GridSampler>(0.2, 0.8, 1.2, 0.1);
	tracker = auto_ptr<CondensationTracker>(new CondensationTracker(resamplingSampler, measurementModel,
			make_shared<FilteringPositionExtractor>(make_shared<WeightedMeanPositionExtractor>())));
}

void FaceTracking::initGui() {
	drawSamples = true;

	cvNamedWindow(controlWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(controlWindowName.c_str(), 750, 50);

	cv::createTrackbar("Grid/Resampling", controlWindowName, NULL, 1, samplerChanged, this);
	cv::setTrackbarPos("Grid/Resampling", controlWindowName, tracker->getSampler() == gridSampler ? 0 : 1);

	cv::createTrackbar("Self-learning active", controlWindowName, NULL, 1, selfLearningChanged, this);
	cv::setTrackbarPos("Self-learning active", controlWindowName, measurementModel->isSelfLearningActive() ? 1 : 0);

	cv::createTrackbar("Draw samples", controlWindowName, NULL, 1, drawSamplesChanged, this);
	cv::setTrackbarPos("Draw samples", controlWindowName, drawSamples ? 1 : 0);
}

void FaceTracking::samplerChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->tracker->setSampler(state == 0 ? tracking->gridSampler : tracking->resamplingSampler);
}

void FaceTracking::selfLearningChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->measurementModel->setSelfLearningActive(state == 1);
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
			cv::Scalar& color = (sit->isObject() > 0.5) ? red : black;
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

	cv::VideoCapture capture;
	if (cam) {
		capture.open(device);
		if (!capture.isOpened()) {
			std::cerr << "Could not open stream from device " << device << std::endl;
			return;
		}
		if (!capture.set(CV_CAP_PROP_FRAME_WIDTH, frameWidth)
				|| !capture.set(CV_CAP_PROP_FRAME_HEIGHT, frameHeight))
			std::cerr << "Could not change resolution to " << frameWidth << "x" << frameHeight << std::endl;
		frameWidth = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
		frameHeight = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	} else {
		capture.open(video);
		if (!capture.isOpened()) {
			std::cerr << "Could not open video file '" << video << "'" << std::endl;
			return;
		}
		frameWidth = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
		frameHeight = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	}

	bool first = true;
	cv::Mat frame, image;
	cv::Scalar green(0, 255, 0); // blue, green, red

	timeval start, detStart, detEnd, frameStart, frameEnd;
	float allDetTimeS = 0;

	int frames = 0;
	gettimeofday(&start, 0);
	std::cout.precision(2);

	while (running) {
		/*while (!commandQueue.empty()) {
			TrackerCommand *command = commandQueue.front();
			commandQueue.pop();
			command->run(trackerCtrl);
			delete command;
		}*/

		frames++;
		gettimeofday(&frameStart, 0);
		capture >> frame;

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
			boost::optional<Rectangle> face = tracker->process(myImage);
			delete myImage;
			gettimeofday(&detEnd, 0);
			image = frame;
			drawDebug(image);
			if (face)
				cv::rectangle(image, cv::Point(face->getX(), face->getY()),
						cv::Point(face->getX() + face->getWidth(), face->getY() + face->getHeight()), green);
			imshow(videoWindowName, image);
			usleep(10000);

			gettimeofday(&frameEnd, 0);

			int itTimeMs = 1000 * (frameEnd.tv_sec - frameStart.tv_sec) + (frameEnd.tv_usec - frameStart.tv_usec) / 1000;
			int detTimeMs = 1000 * (detEnd.tv_sec - detStart.tv_sec) + (detEnd.tv_usec - detStart.tv_usec) / 1000;
			float allTimeS = 1.0 * (frameEnd.tv_sec - start.tv_sec) + 0.0000001 * (frameEnd.tv_usec - start.tv_usec);
			float allFps = frames / allTimeS;
			allDetTimeS += 0.001 * detTimeMs;
			float detFps = frames / allDetTimeS;
			std::cout << "frame: " << frames << "; time: " << itTimeMs << " ms (" << allFps << " fps); detection: " << detTimeMs << "ms (" << detFps << " fps)" << std::endl;

			int c = cv::waitKey(10);
			if ((char) c == 'q') {
				stop();
			}
		}
	}

	capture.release();
}

void FaceTracking::stop() {
	running = false;
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		std::cout << "Usage: -c device OR -v filename" << std::endl;
		return -1;
	}
	FaceTracking *tracker = NULL;
	if (strcmp("-c", argv[1]) == 0) {
		std::istringstream iss(argv[2]);
		int device;
		iss >> device;
		tracker = new FaceTracking(device);
	} else if (strcmp("-v", argv[1]) == 0) {
		tracker = new FaceTracking(argv[2], false);
	}
	tracker->run();
	delete tracker;
	return 0;
}
