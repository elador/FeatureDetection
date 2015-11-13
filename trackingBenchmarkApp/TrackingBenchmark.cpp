/*
 * TrackingBenchmark.cpp
 *
 *  Created on: 17.02.2014
 *      Author: poschmann
 */

#include "TrackingBenchmark.hpp"
#include "logging/LoggerFactory.hpp"
#include "logging/ConsoleAppender.hpp"
#include "logging/FileAppender.hpp"
#include "imageio/LandmarkSource.hpp"
#include "imageio/BobotLandmarkSource.hpp"
#include "imageio/SingleLandmarkSource.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
#include "imageio/CameraImageSource.hpp"
#include "imageio/VideoImageSource.hpp"
#include "imageio/KinectImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/OrderedLabeledImageSource.hpp"
#include "imageio/RepeatingFileImageSource.hpp"
#include "imageio/OrderedLandmarkSink.hpp"
#include "imageio/BobotLandmarkSink.hpp"
#include "imageio/SingleLandmarkSink.hpp"
#include "imageio/Landmark.hpp"
#include "imageio/RectLandmark.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/UnitNormFilter.hpp"
#include "imageprocessing/ConversionFilter.hpp"
#include "imageprocessing/HaarFeatureFilter.hpp"
#include "imageprocessing/IntegralImageFilter.hpp"
#include "imageprocessing/HistEq64Filter.hpp"
#include "imageprocessing/WhiteningFilter.hpp"
#include "imageprocessing/ZeroMeanUnitVarianceFilter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"
#include "imageprocessing/GradientFilter.hpp"
#include "imageprocessing/GradientMagnitudeFilter.hpp"
#include "imageprocessing/GradientBinningFilter.hpp"
#include "imageprocessing/SpatialHistogramFilter.hpp"
#include "imageprocessing/SpatialPyramidHistogramFilter.hpp"
#include "imageprocessing/HogFilter.hpp"
#include "imageprocessing/PyramidHogFilter.hpp"
#include "imageprocessing/ExtendedHogFilter.hpp"
#include "imageprocessing/IntegralGradientFilter.hpp"
#include "imageprocessing/GradientSumFilter.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/DirectImageFeatureExtractor.hpp"
#include "imageprocessing/FilteringFeatureExtractor.hpp"
#include "imageprocessing/FilteringPyramidFeatureExtractor.hpp"
#include "imageprocessing/PatchResizingFeatureExtractor.hpp"
#include "imageprocessing/IntegralFeatureExtractor.hpp"
#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/RbfKernel.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/HistogramIntersectionKernel.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/AgeBasedExampleManagement.hpp"
#include "classification/ConfidenceBasedExampleManagement.hpp"
#include "classification/UnlimitedExampleManagement.hpp"
#include "classification/FixedTrainableProbabilisticSvmClassifier.hpp"
#include "libsvm/LibSvmClassifier.hpp"
#ifdef WITH_LIBLINEAR_CLASSIFIER
	#include "liblinear/LibLinearClassifier.hpp"
#endif
#include "condensation/ResamplingSampler.hpp"
#include "condensation/GridSampler.hpp"
#include "condensation/LowVarianceSampling.hpp"
#include "condensation/SimpleTransitionModel.hpp"
#include "condensation/OpticalFlowTransitionModel.hpp"
#include "condensation/PositionDependentMeasurementModel.hpp"
#include "condensation/ExtendedHogBasedMeasurementModel.hpp"
#include "condensation/FilteringStateExtractor.hpp"
#include "condensation/WeightedMeanStateExtractor.hpp"
#include "condensation/Sample.hpp"
#include "boost/program_options.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/optional.hpp"
#include "boost/lexical_cast.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <sstream>
#include <algorithm>

using namespace logging;
using namespace classification;
using namespace std::chrono;
using libsvm::LibSvmClassifier;
using cv::Point;
using cv::Rect;
using cv::Rect_;
using boost::property_tree::info_parser::read_info;
using boost::lexical_cast;
using std::milli;
using std::move;
using std::ofstream;
using std::ostringstream;
using std::istringstream;
using std::runtime_error;
using std::invalid_argument;

TrackingBenchmark::TrackingBenchmark(ptree& config) {
	initTracking(config);
}

shared_ptr<DirectPyramidFeatureExtractor> TrackingBenchmark::createPyramidExtractor(
		ptree& config, shared_ptr<ImagePyramid> pyramid, bool needsLayerFilters) {
	if (config.get_value<string>() == "direct") {
		shared_ptr<DirectPyramidFeatureExtractor> pyramidExtractor = make_shared<DirectPyramidFeatureExtractor>(
				config.get<int>("patch.width"), config.get<int>("patch.height"),
				config.get<int>("patch.minWidth"), config.get<int>("patch.maxWidth"),
				config.get<int>("interval"));
		pyramidExtractor->addImageFilter(make_shared<GrayscaleFilter>());
		return pyramidExtractor;
	} else if (config.get_value<string>() == "derived") {
		if (!pyramid)
			throw invalid_argument("base pyramid necessary for creating a derived pyramid");
		if (needsLayerFilters)
			return make_shared<DirectPyramidFeatureExtractor>(make_shared<ImagePyramid>(pyramid),
					config.get<int>("patch.width"), config.get<int>("patch.height"));
		else
			return make_shared<DirectPyramidFeatureExtractor>(pyramid,
					config.get<int>("patch.width"), config.get<int>("patch.height"));
	} else {
		throw invalid_argument("invalid pyramid type: " + config.get_value<string>());
	}
}

shared_ptr<FeatureExtractor> TrackingBenchmark::createFeatureExtractor(
		shared_ptr<ImagePyramid> pyramid, ptree& config) {
	float scaleFactor = config.get<float>("scale", 1.f);
	if (config.get_value<string>() == "histeq") {
		pyramidExtractor = createPyramidExtractor(config.get_child("pyramid"), pyramid, false);
		pyramidExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
		return wrapFeatureExtractor(pyramidExtractor, scaleFactor);
	} else if (config.get_value<string>() == "whi") {
		pyramidExtractor = createPyramidExtractor(config.get_child("pyramid"), pyramid, true);
		pyramidExtractor->addPatchFilter(make_shared<WhiteningFilter>());
		pyramidExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
		pyramidExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0 / 127.5, -1.0));
		pyramidExtractor->addPatchFilter(make_shared<UnitNormFilter>(cv::NORM_L2));
		return wrapFeatureExtractor(pyramidExtractor, scaleFactor);
	} else if (config.get_value<string>() == "haar") {
		vector<float> sizes;
		float size;
		istringstream sizesStream(config.get<string>("sizes"));
		while (sizesStream.good() && !sizesStream.fail()) {
			sizesStream >> size;
			sizes.push_back(size);
		}
		float gridRows = config.get<float>("gridRows");
		float gridCols = config.get<float>("gridCols");
		int types = 0;
		string type;
		istringstream typesStream(config.get<string>("types"));
		while (typesStream.good() && !typesStream.fail()) {
			typesStream >> type;
			if (type == "2rect")
				types |= HaarFeatureFilter::TYPE_2RECTANGLE;
			else if (type == "3rect")
				types |= HaarFeatureFilter::TYPE_3RECTANGLE;
			else if (type == "4rect")
				types |= HaarFeatureFilter::TYPE_4RECTANGLE;
			else if (type == "center-surround")
				types |= HaarFeatureFilter::TYPE_CENTER_SURROUND;
			else if (type == "all")
				types |= HaarFeatureFilter::TYPES_ALL;
		}
		shared_ptr<DirectImageFeatureExtractor> featureExtractor = make_shared<DirectImageFeatureExtractor>();
		featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
		featureExtractor->addImageFilter(make_shared<IntegralImageFilter>());
		featureExtractor->addPatchFilter(make_shared<HaarFeatureFilter>(sizes, gridRows, gridCols, types));
		return wrapFeatureExtractor(featureExtractor, scaleFactor);
	} else if (config.get_value<string>() == "hog") {
		pyramidExtractor = createPyramidExtractor(config.get_child("pyramid"), pyramid, true);
		pyramidExtractor->addLayerFilter(make_shared<GradientFilter>(config.get<int>("gradientKernel"), config.get<int>("blurKernel")));
		pyramidExtractor->addLayerFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed")));
		pyramidExtractor->addPatchFilter(createHogFilter(config.get<int>("bins"), config.get_child("histogram")));
		return wrapFeatureExtractor(pyramidExtractor, scaleFactor);
	} else if (config.get_value<string>() == "ihog") {
		shared_ptr<DirectImageFeatureExtractor> featureExtractor = make_shared<DirectImageFeatureExtractor>();
		featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
		featureExtractor->addImageFilter(make_shared<IntegralImageFilter>());
		featureExtractor->addPatchFilter(make_shared<IntegralGradientFilter>(config.get<int>("gradientCount")));
		featureExtractor->addPatchFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed")));
		featureExtractor->addPatchFilter(createHogFilter(config.get<int>("bins"), config.get_child("histogram")));
		return wrapFeatureExtractor(make_shared<IntegralFeatureExtractor>(featureExtractor), scaleFactor);
	} else if (config.get_value<string>() == "ehog") {
		pyramidExtractor = createPyramidExtractor(
				config.get_child("pyramid"), pyramid, true);
		pyramidExtractor->addLayerFilter(make_shared<GradientFilter>(config.get<int>("gradientKernel"), config.get<int>("blurKernel")));
		pyramidExtractor->addLayerFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed")));
		pyramidExtractor->addPatchFilter(make_shared<ExtendedHogFilter>(config.get<int>("bins"), config.get<int>("histogram.cellSize"),
				config.get<bool>("histogram.interpolate"), config.get<bool>("histogram.signedAndUnsigned"), config.get<float>("histogram.alpha")));
		return wrapFeatureExtractor(pyramidExtractor, scaleFactor);
	} else if (config.get_value<string>() == "iehog") {
		shared_ptr<DirectImageFeatureExtractor> featureExtractor = make_shared<DirectImageFeatureExtractor>();
		featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
		featureExtractor->addImageFilter(make_shared<IntegralImageFilter>());
		featureExtractor->addPatchFilter(make_shared<IntegralGradientFilter>(config.get<int>("gradientCount")));
		featureExtractor->addPatchFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed")));
		featureExtractor->addPatchFilter(make_shared<ExtendedHogFilter>(config.get<int>("bins"), config.get<int>("histogram.cellSize"),
				config.get<bool>("histogram.interpolate"), config.get<bool>("histogram.signedAndUnsigned"), config.get<float>("histogram.alpha")));
		return wrapFeatureExtractor(make_shared<IntegralFeatureExtractor>(featureExtractor), scaleFactor);
	} else if (config.get_value<string>() == "surf") {
		shared_ptr<DirectImageFeatureExtractor> featureExtractor = make_shared<DirectImageFeatureExtractor>();
		featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
		featureExtractor->addImageFilter(make_shared<IntegralImageFilter>());
		featureExtractor->addPatchFilter(make_shared<IntegralGradientFilter>(config.get<int>("gradientCount")));
		featureExtractor->addPatchFilter(make_shared<GradientSumFilter>(config.get<int>("cellCount")));
		featureExtractor->addPatchFilter(make_shared<UnitNormFilter>(cv::NORM_L2));
		return wrapFeatureExtractor(make_shared<IntegralFeatureExtractor>(featureExtractor), scaleFactor);
	} else if (config.get_value<string>() == "lbp") {
		pyramidExtractor = createPyramidExtractor(config.get_child("pyramid"), pyramid, true);
		shared_ptr<LbpFilter> lbpFilter = createLbpFilter(config.get<string>("type"));
		pyramidExtractor->addLayerFilter(lbpFilter);
		pyramidExtractor->addPatchFilter(createHistogramFilter(lbpFilter->getBinCount(), config.get_child("histogram")));
		return wrapFeatureExtractor(pyramidExtractor, scaleFactor);
	} else if (config.get_value<string>() == "glbp") {
		pyramidExtractor = createPyramidExtractor(config.get_child("pyramid"), pyramid, true);
		shared_ptr<LbpFilter> lbpFilter = createLbpFilter(config.get<string>("type"));
		pyramidExtractor->addLayerFilter(make_shared<GradientFilter>(config.get<int>("gradientKernel"), config.get<int>("blurKernel")));
		pyramidExtractor->addLayerFilter(make_shared<GradientMagnitudeFilter>());
		pyramidExtractor->addLayerFilter(lbpFilter);
		pyramidExtractor->addPatchFilter(createHistogramFilter(lbpFilter->getBinCount(), config.get_child("histogram")));
		return wrapFeatureExtractor(pyramidExtractor, scaleFactor);
	} else {
		throw invalid_argument("invalid feature type: " + config.get_value<string>());
	}
}

shared_ptr<ImageFilter> TrackingBenchmark::createHogFilter(int bins, ptree& config) {
	if (config.get_value<string>() == "spatial") {
		if (config.get<int>("blockSize") == 1 && !config.get<bool>("signedAndUnsigned"))
			return createHistogramFilter(bins, config);
		else
			return make_shared<HogFilter>(bins, config.get<int>("cellSize"), config.get<int>("blockSize"),
					config.get<bool>("interpolate"), config.get<bool>("signedAndUnsigned"));
	} else if (config.get_value<string>() == "pyramid") {
		return make_shared<PyramidHogFilter>(bins, config.get<int>("levels"), config.get<bool>("interpolate"), config.get<bool>("signedAndUnsigned"));
	} else {
		throw invalid_argument("invalid histogram type: " + config.get_value<string>());
	}
}

shared_ptr<LbpFilter> TrackingBenchmark::createLbpFilter(string lbpType) {
	LbpFilter::Type type;
	if (lbpType == "lbp8")
		type = LbpFilter::Type::LBP8;
	else if (lbpType == "lbp8uniform")
		type = LbpFilter::Type::LBP8_UNIFORM;
	else if (lbpType == "lbp4")
		type = LbpFilter::Type::LBP4;
	else if (lbpType == "lbp4rotated")
		type = LbpFilter::Type::LBP4_ROTATED;
	else
		throw invalid_argument("invalid LBP type: " + lbpType);
	return make_shared<LbpFilter>(type);
}

shared_ptr<HistogramFilter> TrackingBenchmark::createHistogramFilter(unsigned int bins, ptree& config) {
	HistogramFilter::Normalization normalization;
	if (config.get<string>("normalization") == "none")
		normalization = HistogramFilter::Normalization::NONE;
	else if (config.get<string>("normalization") == "l2norm")
		normalization = HistogramFilter::Normalization::L2NORM;
	else if (config.get<string>("normalization") == "l2hys")
		normalization = HistogramFilter::Normalization::L2HYS;
	else if (config.get<string>("normalization") == "l1norm")
		normalization = HistogramFilter::Normalization::L1NORM;
	else if (config.get<string>("normalization") == "l1sqrt")
		normalization = HistogramFilter::Normalization::L1SQRT;
	else
		throw invalid_argument("invalid normalization method: " + config.get<string>("normalization"));
	if (config.get_value<string>() == "spatial")
		return make_shared<SpatialHistogramFilter>(bins, config.get<int>("cellSize"), config.get<int>("blockSize"),
				config.get<bool>("interpolate"), config.get<bool>("concatenate"), normalization);
	else 	if (config.get_value<string>() == "pyramid")
		return make_shared<SpatialPyramidHistogramFilter>(bins, config.get<int>("levels"), config.get<bool>("interpolate"), normalization);
	else
		throw invalid_argument("invalid histogram type: " + config.get_value<string>());
}

shared_ptr<FeatureExtractor> TrackingBenchmark::wrapFeatureExtractor(shared_ptr<FeatureExtractor> featureExtractor, float scaleFactor) {
	if (scaleFactor == 1.0)
		return featureExtractor;
	return make_shared<PatchResizingFeatureExtractor>(featureExtractor, scaleFactor);
}

shared_ptr<Kernel> TrackingBenchmark::createKernel(ptree& config) {
	if (config.get_value<string>() == "rbf") {
		return make_shared<RbfKernel>(config.get<double>("gamma"));
	} else if (config.get_value<string>() == "poly") {
		return make_shared<PolynomialKernel>(
				config.get<double>("alpha"), config.get<double>("constant"), config.get<double>("degree"));
	} else if (config.get_value<string>() == "hik") {
		return make_shared<HistogramIntersectionKernel>();
	} else if (config.get_value<string>() == "linear") {
		return make_shared<LinearKernel>();
	} else {
		throw invalid_argument("invalid kernel type: " + config.get_value<string>());
	}
}

unique_ptr<ExampleManagement> TrackingBenchmark::createExampleManagement(ptree& config, shared_ptr<BinaryClassifier> classifier, bool positive) {
	if (config.get_value<string>() == "unlimited") {
		return unique_ptr<ExampleManagement>(new UnlimitedExampleManagement(config.get<size_t>("required")));
	} else if (config.get_value<string>() == "agebased") {
		return unique_ptr<ExampleManagement>(new AgeBasedExampleManagement(config.get<size_t>("capacity"), config.get<size_t>("required")));
	} else if (config.get_value<string>() == "confidencebased") {
		return unique_ptr<ExampleManagement>(new ConfidenceBasedExampleManagement(classifier, positive, config.get<size_t>("capacity"), config.get<size_t>("required")));
	} else {
		throw invalid_argument("invalid example management type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableSvmClassifier> TrackingBenchmark::createLibSvmClassifier(ptree& config, shared_ptr<Kernel> kernel) {
	if (config.get_value<string>() == "binary") {
		shared_ptr<LibSvmClassifier> trainableSvm = LibSvmClassifier::createBinarySvm(kernel, config.get<double>("C"));
		trainableSvm->setPositiveExampleManagement(
				unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("positiveExamples"), trainableSvm, true)));
		trainableSvm->setNegativeExampleManagement(
				unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("negativeExamples"), trainableSvm, false)));
		optional<ptree&> negativesConfig = config.get_child_optional("staticNegativeExamples");
		if (negativesConfig && negativesConfig->get_value<bool>()) {
			trainableSvm->loadStaticNegatives(negativesConfig->get<string>("filename"),
					negativesConfig->get<int>("amount"), negativesConfig->get<double>("scale"));
		}
		return trainableSvm;
	} else if (config.get_value<string>() == "one-class") {
		shared_ptr<LibSvmClassifier> trainableSvm = LibSvmClassifier::createOneClassSvm(kernel, config.get<double>("nu"));
		trainableSvm->setPositiveExampleManagement(
				unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("positiveExamples"), trainableSvm, true)));
		return trainableSvm;
	} else {
		throw invalid_argument("invalid libSVM training type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableSvmClassifier> TrackingBenchmark::createLibLinearClassifier(ptree& config) {
#ifdef WITH_LIBLINEAR_CLASSIFIER
	shared_ptr<liblinear::LibLinearClassifier> trainableSvm = make_shared<liblinear::LibLinearClassifier>(
			config.get<double>("C"), config.get<bool>("bias"));
	trainableSvm->setPositiveExampleManagement(
			unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("positiveExamples"), trainableSvm, true)));
	trainableSvm->setNegativeExampleManagement(
			unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("negativeExamples"), trainableSvm, false)));
	optional<ptree&> negativesConfig = config.get_child_optional("staticNegativeExamples");
	if (negativesConfig && negativesConfig->get_value<bool>()) {
		trainableSvm->loadStaticNegatives(negativesConfig->get<string>("filename"),
				negativesConfig->get<int>("amount"), negativesConfig->get<double>("scale"));
	}
	return trainableSvm;
#else
	throw std::runtime_error("Cannot load a LibLinear classifier. Run CMake with WITH_LIBLINEAR_CLASSIFIER set to ON to enable.");
#endif // WITH_LIBLINEAR_CLASSIFIER
}

shared_ptr<TrainableProbabilisticClassifier> TrackingBenchmark::createTrainableProbabilisticClassifier(ptree& config) {
	if (config.get_value<string>() == "libSvm") {
		shared_ptr<Kernel> kernel = createKernel(config.get_child("kernel"));
		shared_ptr<TrainableSvmClassifier> trainableSvm = createLibSvmClassifier(config.get_child("training"), kernel);
		shared_ptr<SvmClassifier> svm = trainableSvm->getSvm();
		optional<ptree&> thresholdConfig = config.get_child_optional("threshold");
		if (thresholdConfig)
			svm->setThreshold(thresholdConfig->get_value<float>());
		return createTrainableProbabilisticSvm(trainableSvm, config.get_child("probabilistic"));
	} else if (config.get_value<string>() == "libLinear") {
		shared_ptr<TrainableSvmClassifier> trainableSvm = createLibLinearClassifier(config.get_child("training"));
		shared_ptr<SvmClassifier> svm = trainableSvm->getSvm();
		optional<ptree&> thresholdConfig = config.get_child_optional("threshold");
		if (thresholdConfig)
			svm->setThreshold(thresholdConfig->get_value<float>());
		return createTrainableProbabilisticSvm(trainableSvm, config.get_child("probabilistic"));
	} else {
		throw invalid_argument("invalid classifier type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableProbabilisticClassifier> TrackingBenchmark::createTrainableProbabilisticSvm(
		shared_ptr<TrainableSvmClassifier> trainableSvm, ptree& config) {
	shared_ptr<TrainableProbabilisticSvmClassifier> svm;
	if (config.get_value<string>() == "default")
		svm = make_shared<TrainableProbabilisticSvmClassifier>(trainableSvm,
				config.get<int>("positiveExamples"), config.get<int>("negativeExamples"),
				config.get<double>("positiveProbability"), config.get<double>("negativeProbability"));
	else if (config.get_value<string>() == "precomputed")
		svm = make_shared<FixedTrainableProbabilisticSvmClassifier>(trainableSvm,
				config.get<double>("positiveProbability"), config.get<double>("negativeProbability"),
				config.get<double>("positiveMean"), config.get<double>("negativeMean"));
	else if (config.get_value<string>() == "predefined")
		svm = make_shared<FixedTrainableProbabilisticSvmClassifier>(trainableSvm,
				config.get<double>("logisticA"), config.get<double>("logisticB"));
	else
		throw invalid_argument("invalid probabilistic SVM type: " + config.get_value<string>());
	if (config.get<string>("adjustThreshold") != "no")
		svm->setAdjustThreshold(config.get<double>("adjustThreshold"));
	return svm;
}

void TrackingBenchmark::initTracking(ptree& config) {
	// create base pyramid
	shared_ptr<ImagePyramid> pyramid;
	optional<ptree&> pyramidConfig = config.get_child_optional("pyramid");
	if (pyramidConfig) {
		DirectPyramidFeatureExtractor tmp(
				pyramidConfig->get<int>("patch.width"), pyramidConfig->get<int>("patch.height"),
				pyramidConfig->get<int>("patch.minWidth"), pyramidConfig->get<int>("patch.maxWidth"),
				pyramidConfig->get<int>("interval"));
		tmp.addImageFilter(make_shared<GrayscaleFilter>());
		pyramid = tmp.getPyramid();
	}

	// create adaptive measurement model
	shared_ptr<AdaptiveMeasurementModel> measurementModel;
	shared_ptr<TrainableProbabilisticClassifier> classifier = createTrainableProbabilisticClassifier(config.get_child("adaptive.measurement.classifier"));
	if (config.get<string>("adaptive.measurement") == "ehog") {
		shared_ptr<TrainableProbabilisticSvmClassifier> svmClassifier = std::dynamic_pointer_cast<TrainableProbabilisticSvmClassifier>(classifier);
		if (!svmClassifier)
			throw invalid_argument("AdaptiveTracking: extended HOG based measurement model (ehog) needs a SVM classifier");
		shared_ptr<ExtendedHogBasedMeasurementModel> model;
		if (config.get<string>("adaptive.measurement.pyramid") == "direct")
			model = make_shared<ExtendedHogBasedMeasurementModel>(svmClassifier);
		else if (config.get<string>("adaptive.measurement.pyramid") == "derived")
			model = make_shared<ExtendedHogBasedMeasurementModel>(svmClassifier, pyramid);
		else
			throw invalid_argument("AdaptiveTracking: invalid pyramid type: " + config.get<string>("adaptive.measurement.pyramid"));
		model->setHogParams(
				config.get<size_t>("adaptive.measurement.cellSize"),
				config.get<size_t>("adaptive.measurement.cellCount"),
				config.get<bool>("adaptive.measurement.signedAndUnsigned"),
				config.get<bool>("adaptive.measurement.interpolateBins"),
				config.get<bool>("adaptive.measurement.interpolateCells"),
				config.get<int>("adaptive.measurement.octaveLayerCount"));
		model->setRejectionThreshold(
				config.get<double>("adaptive.measurement.rejectionThreshold"));
		model->setUseSlidingWindow(
				config.get<bool>("adaptive.measurement.useSlidingWindow"),
				config.get<bool>("adaptive.measurement.conservativeReInit"));
		model->setNegativeExampleParams(
				config.get<size_t>("adaptive.measurement.negativeExampleCount"),
				config.get<size_t>("adaptive.measurement.initialNegativeExampleCount"),
				config.get<size_t>("adaptive.measurement.randomExampleCount"),
				config.get<float>("adaptive.measurement.negativeScoreThreshold"));
		model->setOverlapThresholds(
				config.get<double>("adaptive.measurement.positiveOverlapThreshold"),
				config.get<double>("adaptive.measurement.negativeOverlapThreshold"));
		ExtendedHogBasedMeasurementModel::Adaptation adaptation;
		if (config.get<string>("adaptive.measurement.adaptation") == "NONE")
			adaptation = ExtendedHogBasedMeasurementModel::Adaptation::NONE;
		else if (config.get<string>("adaptive.measurement.adaptation") == "POSITION")
			adaptation = ExtendedHogBasedMeasurementModel::Adaptation::POSITION;
		else if (config.get<string>("adaptive.measurement.adaptation") == "TRAJECTORY")
			adaptation = ExtendedHogBasedMeasurementModel::Adaptation::TRAJECTORY;
		else if (config.get<string>("adaptive.measurement.adaptation") == "CORRECTED_TRAJECTORY")
			adaptation = ExtendedHogBasedMeasurementModel::Adaptation::CORRECTED_TRAJECTORY;
		else
			throw invalid_argument("AdaptiveTracking: invalid adaptation type: " + config.get<string>("adaptive.measurement.adaptation"));
		model->setAdaptation(adaptation,
				config.get<double>("adaptive.measurement.adaptationThreshold"),
				config.get<double>("adaptive.measurement.exclusionThreshold"));
		measurementModel = model;
		hogModel = model;
	} else if (config.get<string>("adaptive.measurement") == "positionDependent") {
		shared_ptr<FeatureExtractor> adaptiveFeatureExtractor = createFeatureExtractor(pyramid, config.get_child("adaptive.measurement.feature"));
		shared_ptr<PositionDependentMeasurementModel> model = make_shared<PositionDependentMeasurementModel>(adaptiveFeatureExtractor, classifier);
		model->setFrameCounts(
				config.get<unsigned int>("adaptive.measurement.startFrameCount"),
				config.get<unsigned int>("adaptive.measurement.stopFrameCount"));
		model->setThresholds(
				config.get<float>("adaptive.measurement.targetThreshold"),
				config.get<float>("adaptive.measurement.confidenceThreshold"));
		model->setOffsetFactors(
				config.get<float>("adaptive.measurement.positiveOffsetFactor"),
				config.get<float>("adaptive.measurement.negativeOffsetFactor"));
		model->setSamplingProperties(
				config.get<unsigned int>("adaptive.measurement.sampleNegativesAroundTarget"),
				config.get<unsigned int>("adaptive.measurement.sampleAdditionalNegatives"),
				config.get<unsigned int>("adaptive.measurement.sampleTestNegatives"),
				config.get<bool>("adaptive.measurement.exploitSymmetry"));
		measurementModel = model;
	} else {
		throw invalid_argument("AdaptiveTracking: invalid adaptive measurement model type: " + config.get<string>("adaptive.measurement"));
	}

	// create transition model
	shared_ptr<TransitionModel> transitionModel;
	if (config.get<string>("transition") == "simple") {
		transitionModel = make_shared<SimpleTransitionModel>(
				config.get<double>("transition.positionDeviation"), config.get<double>("transition.sizeDeviation"));
	} else if (config.get<string>("transition") == "opticalFlow") {
		shared_ptr<TransitionModel> fallbackModel = make_shared<SimpleTransitionModel>(
				config.get<double>("transition.fallback.positionDeviation"), config.get<double>("transition.fallback.sizeDeviation"));
		transitionModel = make_shared<OpticalFlowTransitionModel>(
				fallbackModel, config.get<double>("transition.positionDeviation"), config.get<double>("transition.sizeDeviation"));
	} else {
		throw invalid_argument("invalid transition model type: " + config.get<string>("transition"));
	}

	// create tracker
	shared_ptr<ResamplingSampler> resamplingSampler = make_shared<ResamplingSampler>(
			config.get<unsigned int>("adaptive.resampling.particleCount"), config.get<double>("adaptive.resampling.randomRate"),
			make_shared<LowVarianceSampling>(), transitionModel,
			config.get<double>("adaptive.resampling.minSize"), config.get<double>("adaptive.resampling.maxSize"));
	shared_ptr<StateExtractor> stateExtractor = make_shared<FilteringStateExtractor>(make_shared<WeightedMeanStateExtractor>());
	tracker = unique_ptr<AdaptiveCondensationTracker>(new AdaptiveCondensationTracker(
			resamplingSampler, measurementModel, stateExtractor,
			config.get<unsigned int>("adaptive.resampling.particleCount")));
}

std::pair<double, double> TrackingBenchmark::runTest(shared_ptr<LabeledImageSource> imageSource, shared_ptr<OrderedLandmarkSink> landmarkSink, shared_ptr<OrderedLandmarkSink> learnedSink) {
	steady_clock::time_point start = steady_clock::now();
	duration<double> condensationTime;

	// initialization via ground truth
	if (!imageSource->next())
		throw runtime_error("there are no images in source");
	size_t frames = 1;
	Mat frame = imageSource->getImage();
	if (!imageSource->getLandmarks().isEmpty()) {
		shared_ptr<Landmark> landmark = imageSource->getLandmarks().getLandmark();
		Rect_<float> floatBounds = landmark->getRect();
		Rect bounds(
				Point(cvRound(floatBounds.tl().x), cvRound(floatBounds.tl().y)),
				Point(cvRound(floatBounds.br().x), cvRound(floatBounds.br().y)));
		if (landmark->isVisible() && bounds.x >= 0 && bounds.y >= 0 && bounds.br().x < frame.cols && bounds.br().y < frame.rows) {
			if (pyramidExtractor) {
				double dimension = pyramidExtractor->getPatchWidth() * pyramidExtractor->getPatchHeight();
				float aspectRatio = landmark->getHeight() / landmark->getWidth();
				double patchWidth = sqrt(dimension / aspectRatio);
				double patchHeight = aspectRatio * patchWidth;
				pyramidExtractor->setPatchSize(cvRound(patchWidth), cvRound(patchHeight));
			}
			steady_clock::time_point condensationStart = steady_clock::now();
			optional<Rect> position = tracker->initialize(frame, bounds);
			if (!position)
				throw runtime_error("Adaptive tracker could not be initialized with " + lexical_cast<string>(bounds));
			steady_clock::time_point condensationEnd = steady_clock::now();
			condensationTime += duration_cast<milliseconds>(condensationEnd - condensationStart);
			LandmarkCollection collection;
			collection.insert(make_shared<RectLandmark>("target", *position));
			landmarkSink->add(collection);
		}
	}

	// adaptive tracking
	while (imageSource->next() && !imageSource->getLandmarks().isEmpty()) {
		frames++;
		frame = imageSource->getImage();
		steady_clock::time_point condensationStart = steady_clock::now();
		optional<Rect> position = tracker->process(frame);
		steady_clock::time_point condensationEnd = steady_clock::now();
		condensationTime += duration_cast<milliseconds>(condensationEnd - condensationStart);
		LandmarkCollection collection;
		if (position)
			collection.insert(make_shared<RectLandmark>("target", *position));
		else
			collection.insert(make_shared<RectLandmark>("target"));
		landmarkSink->add(collection);
	}
	if (hogModel) {
		const std::unordered_map<size_t, cv::Rect>& positiveExamples = hogModel->getLearned();
		for (size_t i = 0; i < frames; ++i) {
			LandmarkCollection collection;
			auto it = positiveExamples.find(i);
			if (it != positiveExamples.end())
				collection.insert(make_shared<RectLandmark>("target", it->second));
			else
				collection.insert(make_shared<RectLandmark>("target"));
			learnedSink->add(collection);
		}
	}
	tracker->reset();

	steady_clock::time_point end = steady_clock::now();
	duration<double> time = duration_cast<milliseconds>(end - start);
	double fps = frames / time.count();
	double condensationFps = frames / condensationTime.count();
	return std::make_pair(fps, condensationFps);
}

double TrackingBenchmark::runTests(const path& resultsDirectory, size_t count, const ptree& testConfig, Logger& log) {
	string testName = testConfig.get_value<string>();
	optional<string> imageDirectory = testConfig.get_optional<string>("directory");
	optional<string> videoFile = testConfig.get_optional<string>("file");
	optional<string> simpleFile = testConfig.get_optional<string>("simple");
	optional<string> bobotFile = testConfig.get_optional<string>("bobot");
	shared_ptr<ImageSource> imageSource;
	if (videoFile && !imageDirectory)
		imageSource = make_shared<VideoImageSource>(*videoFile);
	else if (!videoFile && imageDirectory)
		imageSource = make_shared<DirectoryImageSource>(*imageDirectory);
	else
		throw invalid_argument("either a video file or a directory must be given for test " + testName);
	bool bobot = false;
	if (simpleFile && !bobotFile)
		bobot = false;
	else if (!simpleFile && bobotFile)
		bobot = true;
	else
		throw invalid_argument("either a bobot or a simple ground truth file must be given for test " + testName);
	shared_ptr<LandmarkSource> landmarkSource;
	shared_ptr<OrderedLandmarkSink> landmarkSink;
	shared_ptr<OrderedLandmarkSink> learnedSink;
	path groundTruthFile;
	if (bobot) {
		shared_ptr<BobotLandmarkSource> bobotLandmarkSource = make_shared<BobotLandmarkSource>(*bobotFile, imageSource);
		landmarkSource = bobotLandmarkSource;
		landmarkSink = make_shared<BobotLandmarkSink>(bobotLandmarkSource->getVideoFilename(), imageSource);
		learnedSink = make_shared<BobotLandmarkSink>(bobotLandmarkSource->getVideoFilename(), imageSource);
		groundTruthFile = path(*bobotFile);
	} else { // simple
		landmarkSource = make_shared<SingleLandmarkSource>(*simpleFile);
		landmarkSink = make_shared<SingleLandmarkSink>();
		learnedSink = make_shared<SingleLandmarkSink>();
		groundTruthFile = path(*simpleFile);
	}
	shared_ptr<LabeledImageSource> source = make_shared<OrderedLabeledImageSource>(imageSource, landmarkSource);

	log.info("=======");
	log.info("Running test " + testName + " " + (count == 1 ? "1 time" : (std::to_string(count) + " times")));
	path testDirectory(resultsDirectory.string() + "/" + testName);
	if (!exists(testDirectory))
		create_directory(testDirectory);
	// TODO Boost must be compiled with --std=C++11 in order for the following line to be linkable (Linux & GCC)
	// see http://boost.2283326.n4.nabble.com/Filesystem-problems-with-g-std-c-0x-td2639716.html for further information
//	copy_file(groundTruthFile, testDirectory / "groundtruth");
	size_t skipped = 0;
	std::pair<double, double> fpsSum(0, 0);
	for (size_t i = 0; i < count; ++i) {
		source->reset();
		string outputFilename = testDirectory.string() + "/run" + std::to_string(i);
		string learnedFilename = testDirectory.string() + "/learned" + std::to_string(i);
		if (exists(path(outputFilename))) {
			skipped++;
		} else {
			landmarkSink->open(outputFilename);
			learnedSink->open(learnedFilename);
			try {
				std::pair<double, double> fps = runTest(source, landmarkSink, learnedSink);
				fpsSum.first += fps.first;
				fpsSum.second += fps.second;
			} catch (std::exception& exc) {
				log.error(string("exception on run " + std::to_string(i) + ": ") + exc.what());
			}
			landmarkSink->close();
			learnedSink->close();
		}
	}
	if (skipped == count)
		log.info("skipped all tests (were already done)");
	else if (skipped == 1)
		log.info("skipped 1 test (was already done)");
	else if (skipped > 1)
		log.info("skipped " + std::to_string(skipped) + " tests (were already done)");

	double hitThreshold = 0.5;
	size_t runCount = 0;
	double hitM = 0;
	double hitS = 0;
	double overlapM = 0;
	double overlapS = 0;
	ostringstream hitDetails;
	hitDetails.setf(std::ios_base::fixed, std::ios_base::floatfield);
	hitDetails.precision(1);
	hitDetails << '(';
	ostringstream overlapDetails;
	overlapDetails.setf(std::ios_base::fixed, std::ios_base::floatfield);
	overlapDetails.precision(1);
	overlapDetails << '(';
	for (size_t i = 0; i < count; ++i) {
		string landmarkOutputFilename(testDirectory.string() + "/run" + std::to_string(i));
		shared_ptr<LandmarkSource> landmarkOutput;
		if (bobot) {
			landmarkOutput = make_shared<BobotLandmarkSource>(landmarkOutputFilename, imageSource);
			imageSource->reset();
			imageSource->next();
		} else { // simple
			landmarkOutput = make_shared<SingleLandmarkSource>(landmarkOutputFilename);
		}
		landmarkSource->reset();

		ofstream overlapOutput(testDirectory.string() + "/overlap" + std::to_string(i));
		overlapOutput.setf(std::ios_base::fixed, std::ios_base::floatfield);
		overlapOutput.precision(3);
		size_t frameCount = 0;
		size_t hitCount = 0;
		double overlapSum = 0;
		while (landmarkSource->next()) {
			shared_ptr<Landmark> truth = landmarkSource->getLandmarks().getLandmark();
			double overlap = 0;
			if (landmarkOutput->next()) {
				shared_ptr<Landmark> output = landmarkOutput->getLandmarks().getLandmark();
				overlap = computeOverlap(truth->getRect(), output->getRect());
			}
			overlapOutput << overlap << '\n';
			frameCount++;
			if (overlap >= hitThreshold)
				hitCount++;
			overlapSum += overlap;
		}
		overlapOutput.close();
		double hitScore = static_cast<double>(hitCount) / frameCount;
		double averageOverlap = overlapSum / frameCount;
		if (i > 0)
			hitDetails << ',' << ' ';
		hitDetails << (100 * hitScore) << '%';
		if (i > 0)
			overlapDetails << ',' << ' ';
		overlapDetails << (100 * averageOverlap) << '%';

		runCount++;
		if (runCount == 1) {
			hitM = hitScore;
			hitS = 0;
			overlapM = averageOverlap;
			overlapS = 0;
		} else {
			double newHitM = hitM + (hitScore - hitM) / runCount;
			double newHitS = hitS + (hitScore - hitM) * (hitScore - newHitM);
			hitM = newHitM;
			hitS = newHitS;
			double newOverlapM = overlapM + (averageOverlap - overlapM) / runCount;
			double newOverlapS = overlapS + (averageOverlap - overlapM) * (averageOverlap - newOverlapM);
			overlapM = newOverlapM;
			overlapS = newOverlapS;
		}
	}
	hitDetails << ')';
	overlapDetails << ')';
	double hitScoreMean = hitM;
	double hitScoreDeviation = 0;
	double averageOverlapMean = overlapM;
	double averageOverlapDeviation = 0;
	if (runCount > 1) {
		hitScoreDeviation = std::sqrt(hitS / (runCount - 1));
		averageOverlapDeviation = std::sqrt(overlapS / (runCount - 1));
	}
	ostringstream hitOutput;
	hitOutput.setf(std::ios_base::fixed, std::ios_base::floatfield);
	hitOutput.precision(1);
	hitOutput << "hits: " << (100 * hitScoreMean) << "% / " << (100 * hitScoreDeviation) << "% " << hitDetails.str();
	log.info(hitOutput.str());
	ostringstream overlapOutput;
	overlapOutput.setf(std::ios_base::fixed, std::ios_base::floatfield);
	overlapOutput.precision(1);
	overlapOutput << "overlap: " << (100 * averageOverlapMean) << "% / " << (100 * averageOverlapDeviation) << "% " << overlapDetails.str();
	log.info(overlapOutput.str());
	ostringstream timeOutput;
	timeOutput.setf(std::ios_base::fixed, std::ios_base::floatfield);
	timeOutput.precision(1);
	timeOutput << "speed: " << (fpsSum.first / count) << " fps, " << (fpsSum.second / count) << " fps (condensation only)";
	log.info(timeOutput.str());
	return 100 * averageOverlapMean;
}

double TrackingBenchmark::computeOverlap(Rect_<float> a, Rect_<float> b) const {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	if (unionArea == 0)
		return 0;
	return intersectionArea / unionArea;
}

// returns a path to a non-existing file by adding a number to the given base pathname
path getNonExistingFile(path basename) {
	path file;
	size_t index = 0;
	do {
		file = path(basename.string() + std::to_string(index));
		index++;
	} while (exists(file));
	return file;
}

int main(int argc, char *argv[]) {
	if (argc < 4) {
		std::cout << "Usage: trackingBenchmarkApp directory testconfig algorithmconfig1 [algorithmconfig2 [algorithmconfig3 [...]]]" << std::endl;
		std::cout << "where" << std::endl;
		std::cout << " directory ... directory to write the test results and logs into" << std::endl;
		std::cout << " testconfig ... configuration file of the test sequences to run" << std::endl;
		std::cout << " algorithmconfig# ... configuration file of an algorithm to test" << std::endl;
		return 0;
	}

	// create directory
	path directory(argv[1]);
	if (!exists(directory)) {
		create_directory(directory);
	} else if (is_directory(directory)) {
		std::cout << "directory " + directory.string() + "already exists. use this directory? (j/n): ";
		char c;
		std::cin >> c;
		if (std::tolower(c) != 'j')
			return 0;
	} else {
		throw invalid_argument("a file named " + directory.string() + " prevents creating a directory with that name");
	}

	// create app-global logger
	path appLogFile = getNonExistingFile(directory / "log");
	Logger& appLog = Loggers->getLogger("app");
	appLog.addAppender(make_shared<ConsoleAppender>(LogLevel::Info));
	appLog.addAppender(make_shared<FileAppender>(LogLevel::Info, appLogFile.string()));

	// test algorithms
	ptree testConfig, algorithmConfig;
	read_info(argv[2], testConfig);
	for (int i = 3; i < argc; ++i) {
		read_info(argv[i], algorithmConfig);
		size_t runcount = algorithmConfig.get<size_t>("runcount");
		string name = algorithmConfig.get<string>("name");
		path algorithmDirectory = directory / name;
		if (!exists(algorithmDirectory))
			create_directory(algorithmDirectory);
		else if (!is_directory(algorithmDirectory))
			throw invalid_argument("a file named " + algorithmDirectory.string() + " prevents creating a directory with that name");

		path algorithmLogFile = getNonExistingFile(algorithmDirectory / "log");
		Logger& algorithmLog = Loggers->getLogger(name);
		algorithmLog.addAppender(make_shared<FileAppender>(LogLevel::Info, algorithmLogFile.string()));

		algorithmLog.info("Starting test runs for " + name);
		TrackingBenchmark benchmark(algorithmConfig.get_child("tracking"));
		auto iterators = testConfig.equal_range("test");
		double scoreSum = 0;
		size_t testCount = 0;
		size_t exceptionCount = 0;
		for (auto it = iterators.first; it != iterators.second; ++it) {
			try {
				scoreSum += benchmark.runTests(algorithmDirectory, runcount, it->second, algorithmLog);
				testCount++;
			} catch (std::exception& exc) {
				algorithmLog.error(string("A wild exception appeared: ") + exc.what());
				exceptionCount++;
			}
		}
		appLog.info(name + ": " + std::to_string(scoreSum / testCount) + "%" + (exceptionCount > 0 ? " (" + std::to_string(exceptionCount) + " exceptions)" : ""));
	}
	return 0;
}
