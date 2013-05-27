/*
 * AdaptiveTracking.hpp
 *
 *  Created on: 14.05.2013
 *      Author: poschmann
 */

#ifndef ADAPTIVETRACKING_HPP_
#define ADAPTIVETRACKING_HPP_

#include "imageio/LabeledImageSource.hpp"
#include "imageio/ImageSink.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "classification/Kernel.hpp"
#include "classification/TrainableSvmClassifier.hpp"
#include "classification/TrainableProbabilisticClassifier.hpp"
#include "condensation/CondensationTracker.hpp"
#include "condensation/AdaptiveCondensationTracker.hpp"
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
using namespace imageprocessing;
using namespace condensation;
using namespace classification;
using cv::Mat;
using boost::property_tree::ptree;
using boost::optional;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

class AdaptiveTracking {
public:

	AdaptiveTracking(unique_ptr<LabeledImageSource> imageSource, unique_ptr<ImageSink> imageSink, ptree config);
	virtual ~AdaptiveTracking();

	void run();
	void stop();

private:

	enum Initialization { AUTOMATIC, MANUAL, GROUND_TRUTH };

	static void adaptiveChanged(int state, void* userdata);
	static void positionScatterChanged(int state, void* userdata);
	static void velocityScatterChanged(int state, void* userdata);
	static void initialSamplerChanged(int state, void* userdata);
	static void initialSampleCountChanged(int state, void* userdata);
	static void initialRandomRateChanged(int state, void* userdata);
	static void adaptiveSamplerChanged(int state, void* userdata);
	static void adaptiveSampleCountChanged(int state, void* userdata);
	static void adaptiveRandomRateChanged(int state, void* userdata);
	static void drawSamplesChanged(int state, void* userdata);

	static void onMouse(int event, int x, int y, int, void* userdata);

	shared_ptr<Kernel> createKernel(ptree config);
	shared_ptr<TrainableSvmClassifier> createTrainableSvm(shared_ptr<Kernel> kernel, ptree config);
	shared_ptr<TrainableProbabilisticClassifier> createClassifier(shared_ptr<TrainableSvmClassifier> trainableSvm, ptree config);
	void initTracking(ptree config);
	void initGui();
	void drawDebug(Mat& image, bool usedAdaptive);
	void drawCrosshair(Mat& image);
	void drawBox(Mat& image);
	void drawGroundTruth(Mat& image, const LandmarkCollection& target);
	void drawTarget(Mat& image, optional<Rect> target, optional<Sample> state, bool usedAdaptive = true);

	static const string videoWindowName;
	static const string controlWindowName;

	Mat frame;
	Mat image;
	unique_ptr<LabeledImageSource> imageSource;
	unique_ptr<ImageSink> imageSink;

	int currentX, currentY;
	int storedX, storedY;

	bool running;
	bool paused;
	bool useAdaptive;
	bool adaptiveUsable;
	bool drawSamples;

	Initialization initialization;
	shared_ptr<DirectPyramidFeatureExtractor> patchExtractor;
	unique_ptr<CondensationTracker> initialTracker;
	unique_ptr<AdaptiveCondensationTracker> adaptiveTracker;
	shared_ptr<MeasurementModel> staticMeasurementModel;
	shared_ptr<AdaptiveMeasurementModel> adaptiveMeasurementModel;
	shared_ptr<SimpleTransitionModel> transitionModel;
	shared_ptr<ResamplingSampler> initialResamplingSampler;
	shared_ptr<ResamplingSampler> adaptiveResamplingSampler;
	shared_ptr<GridSampler> gridSampler;
};

#endif /* ADAPTIVETRACKING_HPP_ */
