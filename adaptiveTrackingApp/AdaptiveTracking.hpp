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
#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/LbpFilter.hpp"
#include "imageprocessing/HistogramFilter.hpp"
#include "classification/Kernel.hpp"
#include "classification/TrainableSvmClassifier.hpp"
#include "classification/ExampleManagement.hpp"
#include "classification/TrainableProbabilisticClassifier.hpp"
#include "condensation/CondensationTracker.hpp"
#include "condensation/AdaptiveCondensationTracker.hpp"
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
using cv::Rect;
using boost::property_tree::ptree;
using boost::optional;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

class AdaptiveTracking {
public:

	AdaptiveTracking(unique_ptr<LabeledImageSource> imageSource, unique_ptr<ImageSink> imageSink, ptree& config);
	virtual ~AdaptiveTracking();

	void run();
	void stop();

private:

	enum class Initialization { AUTOMATIC, MANUAL, GROUND_TRUTH };

	static void adaptiveChanged(int state, void* userdata);
	static void positionDeviationChanged(int state, void* userdata);
	static void sizeDeviationChanged(int state, void* userdata);
	static void initialSamplerChanged(int state, void* userdata);
	static void initialSampleCountChanged(int state, void* userdata);
	static void initialRandomRateChanged(int state, void* userdata);
	static void adaptiveSamplerChanged(int state, void* userdata);
	static void adaptiveSampleCountChanged(int state, void* userdata);
	static void adaptiveRandomRateChanged(int state, void* userdata);
	static void drawSamplesChanged(int state, void* userdata);
	static void drawFlowChanged(int state, void* userdata);

	static void onMouse(int event, int x, int y, int, void* userdata);

	shared_ptr<DirectPyramidFeatureExtractor> createPyramidExtractor(
			ptree& config, shared_ptr<ImagePyramid> pyramid, bool needsLayerFilters);
	shared_ptr<FeatureExtractor> createFeatureExtractor(shared_ptr<ImagePyramid> pyramid, ptree& config);
	shared_ptr<ImageFilter> createHogFilter(int bins, ptree& config);
	shared_ptr<LbpFilter> createLbpFilter(string lbpType);
	shared_ptr<HistogramFilter> createHistogramFilter(unsigned int bins, ptree& config);
	shared_ptr<FeatureExtractor> wrapFeatureExtractor(shared_ptr<FeatureExtractor> featureExtractor, float scaleFactor);
	shared_ptr<Kernel> createKernel(ptree& config);
	unique_ptr<ExampleManagement> createExampleManagement(ptree& config, shared_ptr<BinaryClassifier> classifier, bool positive);
	shared_ptr<TrainableSvmClassifier> createLibSvmClassifier(ptree& config, shared_ptr<Kernel> kernel);
	shared_ptr<TrainableSvmClassifier> createLibLinearClassifier(ptree& config);
	shared_ptr<TrainableProbabilisticClassifier> createTrainableProbabilisticClassifier(ptree& config);
	shared_ptr<TrainableProbabilisticClassifier> createTrainableProbabilisticSvm(
			shared_ptr<TrainableSvmClassifier> trainableSvm, ptree& config);
	void initTracking(ptree& config);
	void initGui();
	void drawDebug(Mat& image, bool usedAdaptive);
	void drawCrosshair(Mat& image);
	void drawBox(Mat& image);
	void drawGroundTruth(Mat& image, const LandmarkCollection& target);
	void drawTarget(Mat& image, optional<Rect> target, bool usedAdaptive, bool adapted);

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
	shared_ptr<DirectPyramidFeatureExtractor> pyramidExtractor;
	unique_ptr<CondensationTracker> initialTracker;
	unique_ptr<AdaptiveCondensationTracker> adaptiveTracker;
	shared_ptr<SimpleTransitionModel> simpleTransitionModel;
	shared_ptr<OpticalFlowTransitionModel> opticalFlowTransitionModel;
	shared_ptr<ResamplingSampler> initialResamplingSampler;
	shared_ptr<ResamplingSampler> adaptiveResamplingSampler;
	shared_ptr<GridSampler> gridSampler;
};

#endif /* ADAPTIVETRACKING_HPP_ */
