/*
 * FaceTracking.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef FACETRACKING_H_
#define FACETRACKING_H_

#include "tracking/CondensationTracker.h"
#include "tracking/SelfLearningWvmSvmModel.h"
#include "tracking/FrameBasedSvmTraining.h"
#include "tracking/GridSampler.h"
#include "tracking/SimpleTransitionModel.h"
#include "tracking/ResamplingSampler.h"
#include "opencv2/highgui/highgui.hpp"
#include "boost/shared_ptr.hpp"
#include <string>

class ImageSource;

using std::auto_ptr;
using boost::shared_ptr;
using boost::make_shared;
using namespace tracking;

class FaceTracking {
public:
	explicit FaceTracking(auto_ptr<ImageSource> imageSource);
	virtual ~FaceTracking();

	void run();
	void stop();

private:

	static void selfLearningChanged(int state, void* userdata);
	static void samplerChanged(int state, void* userdata);
	static void sampleCountChanged(int state, void* userdata);
	static void randomRateChanged(int state, void* userdata);
	static void scatterChanged(int state, void* userdata);
	static void drawSamplesChanged(int state, void* userdata);

	void initTracking();
	void initGui();
	void drawDebug(cv::Mat& image);

	static const std::string svmConfigFile;
	static const std::string negativesFile;
	static const std::string videoWindowName;
	static const std::string controlWindowName;

	auto_ptr<ImageSource> imageSource;

	bool running;
	bool paused;
	bool drawSamples;

	auto_ptr<CondensationTracker> tracker;
	shared_ptr<SelfLearningWvmSvmModel> measurementModel;
	shared_ptr<FrameBasedSvmTraining> svmTraining;
	shared_ptr<SimpleTransitionModel> transitionModel;
	shared_ptr<ResamplingSampler> resamplingSampler;
	shared_ptr<GridSampler> gridSampler;
};

#endif /* FACETRACKING_H_ */
