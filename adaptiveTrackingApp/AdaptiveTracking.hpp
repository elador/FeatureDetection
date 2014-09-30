/*
 * AdaptiveTracking.hpp
 *
 *  Created on: 14.05.2013
 *      Author: poschmann
 */

#ifndef ADAPTIVETRACKING_HPP_
#define ADAPTIVETRACKING_HPP_

#include "imageio/AnnotatedImageSource.hpp"
#include "imageio/ImageSink.hpp"
#include "classification/ExampleManagement.hpp"
#include "classification/BinaryClassifier.hpp"
#include "classification/TrainableProbabilisticSvmClassifier.hpp"
#include "condensation/CondensationTracker.hpp"
#include "condensation/ConstantVelocityModel.hpp"
#include "condensation/OpticalFlowTransitionModel.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/property_tree/ptree.hpp"
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

	/**
	 * Constructs a new adaptive tracking application.
	 *
	 * @param[in] imageSource The source for potentially annotated images.
	 * @param[in] imageSink The sink for images with the tracking output (bounding boxes) drawn on them.
	 * @param[in] config The configuration of the tracking application.
	 */
	AdaptiveTracking(unique_ptr<AnnotatedImageSource> imageSource, unique_ptr<ImageSink> imageSink, ptree& config);

	/**
	 * Starts the tracking.
	 */
	void run();

	/**
	 * Stops the tracking.
	 */
	void stop();

private:

	enum class Initialization { MANUAL, GROUND_TRUTH };

	static void positionDeviationChanged(int state, void* userdata);
	static void sizeDeviationChanged(int state, void* userdata);
	static void sampleCountChanged(int state, void* userdata);
	static void randomRateChanged(int state, void* userdata);
	static void drawSamplesChanged(int state, void* userdata);
	static void drawFlowChanged(int state, void* userdata);

	static void onMouse(int event, int x, int y, int, void* userdata);

	unique_ptr<ExampleManagement> createExampleManagement(ptree& config, shared_ptr<BinaryClassifier> classifier, bool positive);
	shared_ptr<TrainableProbabilisticSvmClassifier> createSvm(ptree& config);
	void initTracking(ptree& config);
	void initGui();
	void drawDebug(Mat& image);
	void drawCrosshair(Mat& image);
	void drawBox(Mat& image);
	void drawGroundTruth(Mat& image, const Rect& target);
	void drawTarget(Mat& image, optional<Rect> target, bool adapted);

	static const string videoWindowName;
	static const string controlWindowName;

	Mat frame;
	Mat image;
	unique_ptr<AnnotatedImageSource> imageSource;
	unique_ptr<ImageSink> imageSink;

	int currentX, currentY;
	int storedX, storedY;

	bool running;
	bool paused;
	bool adaptiveUsable;
	bool drawSamples;
	int drawFlow;

	Initialization initialization;
	unique_ptr<CondensationTracker> tracker;
	shared_ptr<ConstantVelocityModel> simpleTransitionModel;
	shared_ptr<OpticalFlowTransitionModel> opticalFlowTransitionModel;
};

#endif /* ADAPTIVETRACKING_HPP_ */
