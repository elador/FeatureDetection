/*
 * FaceTracking.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef FACETRACKING_H_
#define FACETRACKING_H_

#include "imageio/ImageSource.h"
#include "tracking/LearningCondensationTracker.h"
#include "tracking/LearningStrategy.h"
#include "tracking/LearningMeasurementModel.h"
#include "tracking/LibSvmTraining.h"
#include "tracking/SimpleTransitionModel.h"
#include "tracking/ResamplingSampler.h"
#include "tracking/GridSampler.h"
#include "opencv2/highgui/highgui.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include <string>

using std::auto_ptr;
using boost::shared_ptr;
using boost::make_shared;
using namespace tracking;

class FaceTracking {
public:
	explicit FaceTracking(auto_ptr<imageio::ImageSource> imageSource);
	virtual ~FaceTracking();

	void run();
	void stop();

private:

	static void learningChanged(int state, void* userdata);
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

	auto_ptr<imageio::ImageSource> imageSource;

	bool running;
	bool paused;
	bool drawSamples;

	auto_ptr<LearningCondensationTracker> tracker;
	shared_ptr<LearningStrategy> learningStrategy;
	shared_ptr<LearningMeasurementModel> measurementModel;
	shared_ptr<LibSvmTraining> svmTraining;
	shared_ptr<SimpleTransitionModel> transitionModel;
	shared_ptr<ResamplingSampler> resamplingSampler;
	shared_ptr<GridSampler> gridSampler;
};

#endif /* FACETRACKING_H_ */
