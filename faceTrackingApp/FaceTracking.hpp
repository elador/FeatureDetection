/*
 * FaceTracking.hpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef FACETRACKING_HPP_
#define FACETRACKING_HPP_

#include "imageio/ImageSource.hpp"
#include "imageio/ImageSink.hpp"
#include "condensation/CondensationTracker.hpp"
#include "condensation/MeasurementModel.hpp"
#include "condensation/SimpleTransitionModel.hpp"
#include "condensation/ResamplingSampler.hpp"
#include "condensation/GridSampler.hpp"
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

	FaceTracking(unique_ptr<ImageSource> imageSource, unique_ptr<ImageSink> imageSink);
	virtual ~FaceTracking();

	void run();
	void stop();

private:

	static void samplerChanged(int state, void* userdata);
	static void sampleCountChanged(int state, void* userdata);
	static void randomRateChanged(int state, void* userdata);
	static void positionDeviationChanged(int state, void* userdata);
	static void sizeDeviationChanged(int state, void* userdata);
	static void drawSamplesChanged(int state, void* userdata);

	void initTracking();
	void initGui();
	void drawDebug(Mat& image);

	static const string videoWindowName;
	static const string controlWindowName;

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

#endif /* FACETRACKING_HPP_ */
