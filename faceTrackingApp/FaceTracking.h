/*
 * FaceTracking.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef FACETRACKING_H_
#define FACETRACKING_H_

#include "imageio/ImageSource.h"
#include "imageio/ImageSink.h"
#include "classification/LibSvmTraining.h"
#include "tracking/AdaptiveCondensationTracker.h"
#include "tracking/AdaptiveMeasurementModel.h"
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
using namespace classification;

class FaceTracking {
public:
	explicit FaceTracking(auto_ptr<imageio::ImageSource> imageSource, auto_ptr<imageio::ImageSink> imageSink,
			std::string svmConfigFile, std::string negativesFile);
	virtual ~FaceTracking();

	void run();
	void stop();

private:

	static void adaptiveChanged(int state, void* userdata);
	static void samplerChanged(int state, void* userdata);
	static void sampleCountChanged(int state, void* userdata);
	static void randomRateChanged(int state, void* userdata);
	static void scatterChanged(int state, void* userdata);
	static void drawSamplesChanged(int state, void* userdata);

	void initTracking();
	void initGui();
	void drawDebug(cv::Mat& image);

	static const std::string videoWindowName;
	static const std::string controlWindowName;

	const std::string svmConfigFile;
	const std::string negativesFile;

	auto_ptr<imageio::ImageSource> imageSource;
	auto_ptr<imageio::ImageSink> imageSink;

	bool running;
	bool paused;
	bool drawSamples;

	auto_ptr<AdaptiveCondensationTracker> tracker;
	shared_ptr<MeasurementModel> staticMeasurementModel;
	shared_ptr<AdaptiveMeasurementModel> adaptiveMeasurementModel;
	shared_ptr<SimpleTransitionModel> transitionModel;
	shared_ptr<ResamplingSampler> resamplingSampler;
	shared_ptr<GridSampler> gridSampler;
};

#endif /* FACETRACKING_H_ */
