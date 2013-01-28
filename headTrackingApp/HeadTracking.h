/*
 * HeadTracking.h
 *
 *  Created on: 13.11.2012
 *      Author: poschmann
 */

#ifndef HEADTRACKING_H_
#define HEADTRACKING_H_

#include "imageio/ImageSource.h"
#include "imageio/ImageSink.h"
#include "classification/LibSvmTraining.h"
#include "tracking/AdaptiveCondensationTracker.h"
#include "tracking/AdaptiveMeasurementModel.h"
#include "tracking/SimpleTransitionModel.h"
#include "tracking/ResamplingSampler.h"
#include "tracking/GridSampler.h"
#include "opencv2/highgui/highgui.hpp"
#include <memory>
#include <string>

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using namespace tracking;
using namespace classification;

class HeadTracking {
public:
	explicit HeadTracking(unique_ptr<imageio::ImageSource> imageSource, unique_ptr<imageio::ImageSink> imageSink,
			std::string svmConfigFile, std::string negativesFile);
	virtual ~HeadTracking();

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

	unique_ptr<imageio::ImageSource> imageSource;
	unique_ptr<imageio::ImageSink> imageSink;

	bool running;
	bool paused;
	bool drawSamples;

	unique_ptr<AdaptiveCondensationTracker> tracker;
	shared_ptr<MeasurementModel> staticMeasurementModel;
	shared_ptr<AdaptiveMeasurementModel> adaptiveMeasurementModel;
	shared_ptr<SimpleTransitionModel> transitionModel;
	shared_ptr<ResamplingSampler> resamplingSampler;
	shared_ptr<GridSampler> gridSampler;
};

#endif /* HEADTRACKING_H_ */
