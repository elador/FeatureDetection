/*
 * FaceTracking.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef FACETRACKING_H_
#define FACETRACKING_H_

#include "tracking/CondensationTracker.h"
#include "tracking/SelfLearningWvmOeSvmModel.h"
#include "tracking/FrameBasedSvmTraining.h"
#include "tracking/Sampler.h"
#include "opencv2/highgui/highgui.hpp"
#include "boost/shared_ptr.hpp"
#include <string>

using std::auto_ptr;
using boost::shared_ptr;
using boost::make_shared;
using namespace tracking;

class FaceTracking {
public:
	explicit FaceTracking(int device);
	explicit FaceTracking(std::string video, bool realtime = false);
	virtual ~FaceTracking();

	void run();
	void stop();

private:

	static void samplerChanged(int state, void* userdata);
	static void selfLearningChanged(int state, void* userdata);
	static void drawSamplesChanged(int state, void* userdata);

	void init(std::string video, int device, bool cam, bool realtime);
	void initGui();
	void drawDebug(cv::Mat& image);

	static const std::string svmConfigFile;
	static const std::string negativesFile;
	static const std::string videoWindowName;
	static const std::string controlWindowName;

	std::string video;
	int device;
	bool cam;
	bool realtime;

	int frameHeight;
	int frameWidth;

	bool running;
	bool drawSamples;

	auto_ptr<CondensationTracker> tracker;
	shared_ptr<SelfLearningWvmOeSvmModel> measurementModel;
	shared_ptr<FrameBasedSvmTraining> svmTraining;
	shared_ptr<Sampler> resamplingSampler;
	shared_ptr<Sampler> gridSampler;
};

#endif /* FACETRACKING_H_ */
