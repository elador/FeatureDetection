/*
 * PartiallyAdaptiveTracking.hpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#ifndef PARTIALLYADAPTIVETRACKING_HPP_
#define PARTIALLYADAPTIVETRACKING_HPP_

#include "imageio/ImageSource.hpp"
#include "imageio/ImageSink.hpp"
#include "classification/Kernel.hpp"
#include "classification/TrainableSvmClassifier.hpp"
#include "classification/TrainableProbabilisticClassifier.hpp"
#include "condensation/PartiallyAdaptiveCondensationTracker.hpp"
#include "condensation/AdaptiveMeasurementModel.hpp"
#include "condensation/MeasurementModel.hpp"
#include "condensation/SimpleTransitionModel.hpp"
#include "condensation/ResamplingSampler.hpp"
#include "condensation/GridSampler.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/property_tree/ptree.hpp"
#include <memory>
#include <string>
#include <memory>
#include <string>

using namespace imageio;
using namespace condensation;
using namespace classification;
using cv::Mat;
using boost::property_tree::ptree;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

class PartiallyAdaptiveTracking {
public:

	PartiallyAdaptiveTracking(unique_ptr<ImageSource> imageSource, unique_ptr<ImageSink> imageSink, ptree config);
	virtual ~PartiallyAdaptiveTracking();

	void run();
	void stop();

private:

	static void adaptiveChanged(int state, void* userdata);
	static void samplerChanged(int state, void* userdata);
	static void sampleCountChanged(int state, void* userdata);
	static void randomRateChanged(int state, void* userdata);
	static void positionDeviationChanged(int state, void* userdata);
	static void sizeDeviationChanged(int state, void* userdata);
	static void drawSamplesChanged(int state, void* userdata);

	shared_ptr<Kernel> createKernel(ptree config);
	shared_ptr<TrainableSvmClassifier> createTrainableSvm(shared_ptr<Kernel> kernel, ptree config);
	shared_ptr<TrainableProbabilisticClassifier> createClassifier(shared_ptr<TrainableSvmClassifier> trainableSvm, ptree config);
	void initTracking(ptree config);
	void initGui();
	void drawDebug(Mat& image);

	static const string videoWindowName;
	static const string controlWindowName;

	unique_ptr<ImageSource> imageSource;
	unique_ptr<ImageSink> imageSink;

	bool running;
	bool paused;
	bool drawSamples;

	unique_ptr<PartiallyAdaptiveCondensationTracker> tracker;
	shared_ptr<MeasurementModel> staticMeasurementModel;
	shared_ptr<AdaptiveMeasurementModel> adaptiveMeasurementModel;
	shared_ptr<SimpleTransitionModel> transitionModel;
	shared_ptr<ResamplingSampler> resamplingSampler;
	shared_ptr<GridSampler> gridSampler;
};

#endif /* PARTIALLYADAPTIVETRACKING_HPP_ */
