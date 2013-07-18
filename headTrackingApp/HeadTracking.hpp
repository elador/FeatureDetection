/*
 * HeadTracking.hpp
 *
 *  Created on: 14.05.2013
 *      Author: poschmann
 */

#ifndef HEADTRACKING_HPP_
#define HEADTRACKING_HPP_

#include "imageio/LabeledImageSource.hpp"
#include "imageio/ImageSink.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/LbpFilter.hpp"
#include "classification/Kernel.hpp"
#include "classification/TrainableSvmClassifier.hpp"
#include "classification/TrainableProbabilisticClassifier.hpp"
#include "classification/ProbabilisticRvmClassifier.hpp"
#include "condensation/CondensationTracker.hpp"
#include "condensation/AdaptiveCondensationTracker.hpp"
#include "condensation/AdaptiveMeasurementModel.hpp"
#include "condensation/MeasurementModel.hpp"
#include "condensation/SimpleTransitionModel.hpp"
#include "condensation/OpticalFlowTransitionModel.hpp"
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

class HeadTracking {
public:

	HeadTracking(unique_ptr<LabeledImageSource> imageSource, unique_ptr<ImageSink> imageSink, ptree& config);
	~HeadTracking();

	void run();
	void stop();

private:

	enum class Initialization { AUTOMATIC, MANUAL, GROUND_TRUTH };

	static void adaptiveChanged(int state, void* userdata);
	static void scatterChanged(int state, void* userdata);
	static void positionScatterChanged(int state, void* userdata);
	static void velocityScatterChanged(int state, void* userdata);
	static void initialSamplerChanged(int state, void* userdata);
	static void initialSampleCountChanged(int state, void* userdata);
	static void initialRandomRateChanged(int state, void* userdata);
	static void adaptiveSamplerChanged(int state, void* userdata);
	static void adaptiveSampleCountChanged(int state, void* userdata);
	static void adaptiveRandomRateChanged(int state, void* userdata);
	static void numFiltersChanged(int state, void* userdata);
	static void drawSamplesChanged(int state, void* userdata);
	static void drawFlowChanged(int state, void* userdata);

	static void onMouse(int event, int x, int y, int, void* userdata);

	shared_ptr<FeatureExtractor> createFeatureExtractor(shared_ptr<DirectPyramidFeatureExtractor> patchExtractor, ptree& config);
	shared_ptr<LbpFilter> createLbpFilter(string lbpType);
	shared_ptr<FeatureExtractor> createHistogramFeatureExtractor(shared_ptr<FeatureExtractor> patchExtractor, unsigned int bins, ptree& config);
	shared_ptr<FeatureExtractor> wrapFeatureExtractor(shared_ptr<FeatureExtractor> featureExtractor, float scaleFactor);
	shared_ptr<Kernel> createKernel(ptree& config);
	shared_ptr<TrainableSvmClassifier> createTrainableSvm(shared_ptr<Kernel> kernel, ptree& config);
	shared_ptr<TrainableProbabilisticClassifier> createClassifier(shared_ptr<TrainableSvmClassifier> trainableSvm, ptree& config);
	void initTracking(ptree& config);
	void initGui();
	void drawDebug(Mat& image, bool usedAdaptive);
	void drawCrosshair(Mat& image);
	void drawBox(Mat& image);
	void drawTarget(Mat& image, optional<Rect> target, optional<Sample> state, bool usedAdaptive, bool adapted);

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
	int drawFlow;

	Initialization initialization;
	shared_ptr<ProbabilisticRvmClassifier> filter;
	shared_ptr<DirectPyramidFeatureExtractor> patchExtractor;
	shared_ptr<FeatureExtractor> adaptiveFeatureExtractor;
	unique_ptr<CondensationTracker> initialTracker;
	unique_ptr<AdaptiveCondensationTracker> adaptiveTracker;
	shared_ptr<MeasurementModel> staticMeasurementModel;
	shared_ptr<AdaptiveMeasurementModel> adaptiveMeasurementModel;
	shared_ptr<SimpleTransitionModel> simpleTransitionModel;
	shared_ptr<OpticalFlowTransitionModel> opticalFlowTransitionModel;
	shared_ptr<ResamplingSampler> initialResamplingSampler;
	shared_ptr<ResamplingSampler> adaptiveResamplingSampler;
	shared_ptr<GridSampler> gridSampler;
};

#endif /* HEADTRACKING_HPP_ */
