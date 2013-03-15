/*
 * AdaptiveTracking.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef ADAPTIVETRACKING_H_
#define ADAPTIVETRACKING_H_

#include "imageio/ImageSource.h"
#include "imageio/ImageSink.h"
#include "condensation/AdaptiveCondensationTracker.h"
#include "condensation/AdaptiveMeasurementModel.h"
#include "condensation/MeasurementModel.h"
#include "condensation/SimpleTransitionModel.h"
#include "condensation/ResamplingSampler.h"
#include "condensation/GridSampler.h"
#include "opencv2/highgui/highgui.hpp"
#include <memory>
#include <string>
#include <memory>
#include <string>

using namespace imageio;
using namespace condensation;
using cv::Mat;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

class AdaptiveTracking {
public:

	AdaptiveTracking(unique_ptr<ImageSource> imageSource, unique_ptr<ImageSink> imageSink, string svmConfigFile, string negativesFile);
	virtual ~AdaptiveTracking();

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
	void drawDebug(Mat& image);

	static const string videoWindowName;
	static const string controlWindowName;

	const string svmConfigFile;
	const string negativesFile;

	unique_ptr<ImageSource> imageSource;
	unique_ptr<ImageSink> imageSink;

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

#endif /* ADAPTIVETRACKING_H_ */
