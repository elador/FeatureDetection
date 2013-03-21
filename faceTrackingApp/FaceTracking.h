/*
 * FaceTracking.h
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef FACETRACKING_H_
#define FACETRACKING_H_

#include "imageio/ImageSource.hpp"
#include "imageio/ImageSink.hpp"
#include "condensation/CondensationTracker.h"
#include "condensation/MeasurementModel.h"
#include "condensation/SimpleTransitionModel.h"
#include "condensation/ResamplingSampler.h"
#include "condensation/GridSampler.h"
#include "opencv2/highgui/highgui.hpp"
#include <memory>
#include <string>

using namespace imageio;
using namespace condensation;
using cv::Mat;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

class FaceTracking {
public:

	FaceTracking(unique_ptr<ImageSource> imageSource, unique_ptr<ImageSink> imageSink, string svmConfigFile, string negativesFile);
	virtual ~FaceTracking();

	void run();
	void stop();

private:

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

	unique_ptr<CondensationTracker> tracker;
	shared_ptr<MeasurementModel> measurementModel;
	shared_ptr<SimpleTransitionModel> transitionModel;
	shared_ptr<ResamplingSampler> resamplingSampler;
	shared_ptr<GridSampler> gridSampler;
};

#endif /* FACETRACKING_H_ */
