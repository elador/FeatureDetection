/*
 * TrackingBenchmark.hpp
 *
 *  Created on: 17.02.2014
 *      Author: poschmann
 */

#ifndef TRACKINGBENCHMARK_HPP_
#define TRACKINGBENCHMARK_HPP_

#include "logging/Logger.hpp"
#include "imageio/LabeledImageSource.hpp"
#include "imageio/OrderedLandmarkSink.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/LbpFilter.hpp"
#include "imageprocessing/HistogramFilter.hpp"
#include "classification/Kernel.hpp"
#include "classification/TrainableSvmClassifier.hpp"
#include "classification/ExampleManagement.hpp"
#include "classification/TrainableProbabilisticClassifier.hpp"
#include "condensation/AdaptiveCondensationTracker.hpp"
#include "condensation/SimpleTransitionModel.hpp"
#include "condensation/OpticalFlowTransitionModel.hpp"
#include "condensation/ResamplingSampler.hpp"
#include "condensation/ExtendedHogBasedMeasurementModel.hpp"
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
using boost::filesystem::path;
using boost::optional;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

class TrackingBenchmark {
public:

	TrackingBenchmark(ptree& config);

	double runTests(const path& resultsDirectory, size_t count, const ptree& config, logging::Logger& log);
	std::pair<double, double> runTest(shared_ptr<LabeledImageSource> imageSource, shared_ptr<OrderedLandmarkSink> landmarkSink, shared_ptr<OrderedLandmarkSink> learnedSink);
	double computeOverlap(cv::Rect_<float> a, cv::Rect_<float> b) const;

private:

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

	shared_ptr<DirectPyramidFeatureExtractor> pyramidExtractor;
	unique_ptr<AdaptiveCondensationTracker> tracker;
	shared_ptr<ExtendedHogBasedMeasurementModel> hogModel;
};

#endif /* TRACKINGBENCHMARK_HPP_ */
