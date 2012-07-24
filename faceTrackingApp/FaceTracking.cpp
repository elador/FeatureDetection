/*
 * FaceTracking.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "FaceTracking.h"
#include "tracking/LowVarianceSampling.h"
#include "tracking/SimpleTransitionModel.h"
#include "tracking/DualSvmMeasurementModel.h"
#include "tracking/WeightedMeanPositionExtractor.h"
#include "tracking/Rectangle.h"
#include "tracking/Sample.h"
#include "VDetectorVectorMachine.h"
#include "DetectorWVM.h"
#include "DetectorSVM.h"
#include "FdImage.h"
#include "boost/optional.hpp"
#include <vector>
#include <iostream>
#include <cstring>
#include <sys/time.h>

const std::string FaceTracking::svmConfigFile = "/home/poschmann/projects/ffd/config/fdetection/fd_config_fft_fd.mat";
const std::string FaceTracking::videoWindowName = "Image";

FaceTracking::FaceTracking(int device) {
	init("", device, true, false);
}

FaceTracking::FaceTracking(std::string video, bool realtime) {
	init(video, 0, false, realtime);
}

FaceTracking::~FaceTracking() {}

void FaceTracking::init(std::string video, int device, bool cam, bool realtime) {
	this->video = video;
	this->device = device;
	this->cam = cam;
	this->realtime = realtime;

	frameWidth = 640;
	frameHeight = 480;
}

CondensationTracker* FaceTracking::createTracker() {
	unsigned int count = 500;
	double randomRate = 0.2;
	VDetectorVectorMachine* wvm = new DetectorWVM();
	wvm->load(svmConfigFile);
	VDetectorVectorMachine* svm = new DetectorSVM();
	svm->load(svmConfigFile);
	return new CondensationTracker(count, randomRate, new LowVarianceSampling(),
			new SimpleTransitionModel(), new DualSvmMeasurementModel(wvm, svm),
			new WeightedMeanPositionExtractor());
}

void FaceTracking::drawDebug(cv::Mat& image, CondensationTracker* tracker) {
	cv::Scalar black(0, 0, 0); // blue, green, red
	cv::Scalar red(0, 0, 255); // blue, green, red
	const std::vector<Sample> samples = tracker->getSamples();
	for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		cv::Scalar& color = (sit->getWeight() > 0.5) ? red : black;
		cv::circle(image, cv::Point(sit->getX(), sit->getY()), 1, color);
//		Rectangle bounds = sit->getBounds();
//		cv::rectangle(image, cv::Point(bounds.getX(), bounds.getY()),
//				cv::Point(bounds.getX() + bounds.getWidth(), bounds.getY() + bounds.getHeight()), color);
	}
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

	CondensationTracker* tracker = createTracker();
	//initGui(tracker);

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
			drawDebug(image, tracker);
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
	delete tracker;
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
