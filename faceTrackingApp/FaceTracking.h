/*
 * FaceTracking.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef FACETRACKING_H_
#define FACETRACKING_H_

#include "tracking/CondensationTracker.h"
#include "opencv2/highgui/highgui.hpp"
#include <string>

using namespace tracking;

class FaceTracking {
public:
	explicit FaceTracking(int device);
	explicit FaceTracking(std::string video, bool realtime = false);
	virtual ~FaceTracking();

	void run();
	void stop();

private:

	void init(std::string video, int device, bool cam, bool realtime);
	CondensationTracker* createTracker();
	void drawDebug(cv::Mat& image, CondensationTracker* tracker);

	static const std::string svmConfigFile;
	static const std::string videoWindowName;

	std::string video;
	int device;
	bool cam;
	bool realtime;

	int frameHeight;
	int frameWidth;

	bool running;
};

#endif /* FACETRACKING_H_ */
