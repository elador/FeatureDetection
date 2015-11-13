/*
 * HeadTracking.cpp
 *
 *  Created on: 14.05.2013
 *      Author: poschmann
 */

#include "HeadTracking.hpp"
#include "logging/LoggerFactory.hpp"
#include "logging/Logger.hpp"
#include "logging/ConsoleAppender.hpp"
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
#include "imageio/VideoImageSink.hpp"
#include "imageio/Landmark.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/HistEq64Filter.hpp"
#include "imageprocessing/IntegralImageFilter.hpp"
#include "imageprocessing/HaarFeatureFilter.hpp"
#include "imageprocessing/WhiteningFilter.hpp"
#include "imageprocessing/ZeroMeanUnitVarianceFilter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"
#include "imageprocessing/GradientFilter.hpp"
#include "imageprocessing/GradientMagnitudeFilter.hpp"
#include "imageprocessing/GradientBinningFilter.hpp"
#include "imageprocessing/ConversionFilter.hpp"
#include "imageprocessing/UnitNormFilter.hpp"
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
#include "classification/ProbabilisticRvmClassifier.hpp"
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
#include "condensation/MeasurementModel.hpp"
#include "condensation/WvmSvmModel.hpp"
#include "condensation/SingleClassifierModel.hpp"
#include "condensation/FilteringClassifierModel.hpp"
#include "condensation/AdaptiveMeasurementModel.hpp"
#include "condensation/SelfLearningMeasurementModel.hpp"
#include "condensation/PositionDependentMeasurementModel.hpp"
#include "condensation/ExtendedHogBasedMeasurementModel.hpp"
#include "condensation/FilteringStateExtractor.hpp"
#include "condensation/WeightedMeanStateExtractor.hpp"
#include "condensation/Sample.hpp"
#include "condensation/ClassificationBasedStateValidator.hpp"
#include "boost/program_options.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/optional.hpp"
#include "boost/tokenizer.hpp"
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
using boost::property_tree::info_parser::read_info;
using std::milli;
using std::move;
using std::ostringstream;
using std::istringstream;
using std::invalid_argument;

namespace po = boost::program_options;

const string HeadTracking::videoWindowName = "Image";
const string HeadTracking::controlWindowName = "Controls";

HeadTracking::HeadTracking(unique_ptr<LabeledImageSource> imageSource, unique_ptr<ImageSink> imageSink, ptree& config) :
		imageSource(move(imageSource)), imageSink(move(imageSink)) {
	initTracking(config);
	initGui();
}

HeadTracking::~HeadTracking() {}

shared_ptr<DirectPyramidFeatureExtractor> HeadTracking::createPyramidExtractor(
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
			throw invalid_argument("HeadTracking: base pyramid necessary for creating a derived pyramid");
		if (needsLayerFilters)
			return make_shared<DirectPyramidFeatureExtractor>(make_shared<ImagePyramid>(pyramid),
					config.get<int>("patch.width"), config.get<int>("patch.height"));
		else
			return make_shared<DirectPyramidFeatureExtractor>(pyramid,
					config.get<int>("patch.width"), config.get<int>("patch.height"));
	} else {
		throw invalid_argument("HeadTracking: invalid pyramid type: " + config.get_value<string>());
	}
}

shared_ptr<FeatureExtractor> HeadTracking::createFeatureExtractor(
		shared_ptr<ImagePyramid> pyramid, ptree& config) {
	float scaleFactor = config.get<float>("scale", 1.f);
	if (config.get_value<string>() == "grayscale") {
		shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = createPyramidExtractor(
				config.get_child("pyramid"), pyramid, false);
		featureExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0 / 255.0));
		return wrapFeatureExtractor(featureExtractor, scaleFactor);
	} else if (config.get_value<string>() == "histeq") {
		shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = createPyramidExtractor(
				config.get_child("pyramid"), pyramid, false);
		featureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
		return wrapFeatureExtractor(featureExtractor, scaleFactor);
	} else if (config.get_value<string>() == "h") {
		shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = createPyramidExtractor(
				config.get_child("pyramid"), pyramid, false);
		featureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
		featureExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0 / 127.5, -1.0));
		return wrapFeatureExtractor(featureExtractor, scaleFactor);
	} else if (config.get_value<string>() == "whi") {
		shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = createPyramidExtractor(
				config.get_child("pyramid"), pyramid, false);
		featureExtractor->addPatchFilter(make_shared<WhiteningFilter>());
		featureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
		featureExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0 / 127.5, -1.0));
		featureExtractor->addPatchFilter(make_shared<UnitNormFilter>(cv::NORM_L2));
		return wrapFeatureExtractor(featureExtractor, scaleFactor);
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
		shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = createPyramidExtractor(
				config.get_child("pyramid"), pyramid, true);
		featureExtractor->addLayerFilter(make_shared<GradientFilter>(config.get<int>("gradientKernel"), config.get<int>("blurKernel")));
		featureExtractor->addLayerFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed"), config.get<bool>("interpolate")));
		featureExtractor->addPatchFilter(createHogFilter(config.get<int>("bins"), config.get_child("histogram")));
		return wrapFeatureExtractor(featureExtractor, scaleFactor);
	} else if (config.get_value<string>() == "ihog") {
		shared_ptr<DirectImageFeatureExtractor> featureExtractor = make_shared<DirectImageFeatureExtractor>();
		featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
		featureExtractor->addImageFilter(make_shared<IntegralImageFilter>());
		featureExtractor->addPatchFilter(make_shared<IntegralGradientFilter>(config.get<int>("gradientCount")));
		featureExtractor->addPatchFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed"), config.get<bool>("interpolate")));
		featureExtractor->addPatchFilter(createHogFilter(config.get<int>("bins"), config.get_child("histogram")));
		return wrapFeatureExtractor(make_shared<IntegralFeatureExtractor>(featureExtractor), scaleFactor);
	} else if (config.get_value<string>() == "ehog") {
		shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = createPyramidExtractor(
				config.get_child("pyramid"), pyramid, true);
		featureExtractor->addLayerFilter(make_shared<GradientFilter>(config.get<int>("gradientKernel"), config.get<int>("blurKernel")));
		featureExtractor->addLayerFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed"), config.get<bool>("interpolate")));
		featureExtractor->addPatchFilter(make_shared<ExtendedHogFilter>(config.get<int>("bins"), config.get<int>("histogram.cellSize"),
				config.get<bool>("histogram.interpolate"), config.get<bool>("histogram.signedAndUnsigned"), config.get<float>("histogram.alpha")));
		return wrapFeatureExtractor(featureExtractor, scaleFactor);
	} else if (config.get_value<string>() == "iehog") {
		shared_ptr<DirectImageFeatureExtractor> featureExtractor = make_shared<DirectImageFeatureExtractor>();
		featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
		featureExtractor->addImageFilter(make_shared<IntegralImageFilter>());
		featureExtractor->addPatchFilter(make_shared<IntegralGradientFilter>(config.get<int>("gradientCount")));
		featureExtractor->addPatchFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed"), config.get<bool>("interpolate")));
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
		shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = createPyramidExtractor(
				config.get_child("pyramid"), pyramid, true);
		shared_ptr<LbpFilter> lbpFilter = createLbpFilter(config.get<string>("type"));
		featureExtractor->addLayerFilter(lbpFilter);
		featureExtractor->addPatchFilter(createHistogramFilter(lbpFilter->getBinCount(), config.get_child("histogram")));
		return wrapFeatureExtractor(featureExtractor, scaleFactor);
	} else if (config.get_value<string>() == "glbp") {
		shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = createPyramidExtractor(
				config.get_child("pyramid"), pyramid, true);
		shared_ptr<LbpFilter> lbpFilter = createLbpFilter(config.get<string>("type"));
		featureExtractor->addLayerFilter(make_shared<GradientFilter>(config.get<int>("gradientKernel"), config.get<int>("blurKernel")));
		featureExtractor->addLayerFilter(make_shared<GradientMagnitudeFilter>());
		featureExtractor->addLayerFilter(lbpFilter);
		featureExtractor->addPatchFilter(createHistogramFilter(lbpFilter->getBinCount(), config.get_child("histogram")));
		return wrapFeatureExtractor(featureExtractor, scaleFactor);
	} else {
		throw invalid_argument("HeadTracking: invalid feature type: " + config.get_value<string>());
	}
}

shared_ptr<ImageFilter> HeadTracking::createHogFilter(int bins, ptree& config) {
	if (config.get_value<string>() == "spatial") {
		if (config.get<int>("blockSize") == 1 && !config.get<bool>("signedAndUnsigned"))
			return createHistogramFilter(bins, config);
		else
			return make_shared<HogFilter>(bins, config.get<int>("cellSize"), config.get<int>("blockSize"),
					config.get<bool>("interpolate"), config.get<bool>("signedAndUnsigned"));
	} else if (config.get_value<string>() == "pyramid") {
		return make_shared<PyramidHogFilter>(bins, config.get<int>("levels"), config.get<bool>("interpolate"), config.get<bool>("signedAndUnsigned"));
	} else {
		throw invalid_argument("HeadTracking: invalid histogram type: " + config.get_value<string>());
	}
}

shared_ptr<LbpFilter> HeadTracking::createLbpFilter(string lbpType) {
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
		throw invalid_argument("HeadTracking: invalid LBP type: " + lbpType);
	return make_shared<LbpFilter>(type);
}

shared_ptr<HistogramFilter> HeadTracking::createHistogramFilter(unsigned int bins, ptree& config) {
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
		throw invalid_argument("HeadTracking: invalid normalization method: " + config.get<string>("normalization"));
	if (config.get_value<string>() == "spatial")
		return make_shared<SpatialHistogramFilter>(bins, config.get<int>("cellSize"), config.get<int>("blockSize"),
				config.get<bool>("interpolate"), config.get<bool>("concatenate"), normalization);
	else 	if (config.get_value<string>() == "pyramid")
		return make_shared<SpatialPyramidHistogramFilter>(bins, config.get<int>("levels"), config.get<bool>("interpolate"), normalization);
	else
		throw invalid_argument("HeadTracking: invalid histogram type: " + config.get_value<string>());
}

shared_ptr<FeatureExtractor> HeadTracking::wrapFeatureExtractor(shared_ptr<FeatureExtractor> featureExtractor, float scaleFactor) {
	if (scaleFactor == 1.0)
		return featureExtractor;
	return make_shared<PatchResizingFeatureExtractor>(featureExtractor, scaleFactor);
}

shared_ptr<Kernel> HeadTracking::createKernel(ptree& config) {
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
		throw invalid_argument("HeadTracking: invalid kernel type: " + config.get_value<string>());
	}
}

unique_ptr<ExampleManagement> HeadTracking::createExampleManagement(ptree& config, shared_ptr<BinaryClassifier> classifier, bool positive) {
	if (config.get_value<string>() == "unlimited") {
		return unique_ptr<ExampleManagement>(new UnlimitedExampleManagement(config.get<size_t>("required")));
	} else if (config.get_value<string>() == "agebased") {
		return unique_ptr<ExampleManagement>(new AgeBasedExampleManagement(config.get<size_t>("capacity"), config.get<size_t>("required")));
	} else if (config.get_value<string>() == "confidencebased") {
		return unique_ptr<ExampleManagement>(new ConfidenceBasedExampleManagement(classifier, positive, config.get<size_t>("capacity"), config.get<size_t>("required")));
	} else {
		throw invalid_argument("HeadTracking: invalid example management type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableSvmClassifier> HeadTracking::createLibSvmClassifier(ptree& config, shared_ptr<Kernel> kernel) {
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
		throw invalid_argument("HeadTracking: invalid libSVM training type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableSvmClassifier> HeadTracking::createLibLinearClassifier(ptree& config) {
#ifdef WITH_LIBLINEAR_CLASSIFIER
	shared_ptr<liblinear::LibLinearClassifier> trainableSvm = make_shared<liblinear::LibLinearClassifier>(config.get<double>("C"), config.get<bool>("bias"));
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

shared_ptr<TrainableProbabilisticClassifier> HeadTracking::createTrainableProbabilisticClassifier(ptree& config) {
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
		throw invalid_argument("HeadTracking: invalid classifier type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableProbabilisticClassifier> HeadTracking::createTrainableProbabilisticSvm(
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
		throw invalid_argument("HeadTracking: invalid probabilistic SVM type: " + config.get_value<string>());
	if (config.get<string>("adjustThreshold") != "no")
		svm->setAdjustThreshold(config.get<double>("adjustThreshold"));
	return svm;
}

/**
 * Extracts double values from a string.
 */
static vector<double> readValues(string text) {
	boost::char_separator<char> sep(" ");
	boost::tokenizer<boost::char_separator<char>> tokens(text, sep);
	vector<double> values;
	for (const string& token : tokens)
		values.push_back(std::stod(token));
	return values;
}

void HeadTracking::initTracking(ptree& config) {
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
	shared_ptr<AdaptiveMeasurementModel> adaptiveMeasurementModel;
	shared_ptr<TrainableProbabilisticClassifier> classifier = createTrainableProbabilisticClassifier(config.get_child("adaptive.measurement.classifier"));
	if (config.get<string>("adaptive.measurement") == "ehog") {
		shared_ptr<TrainableProbabilisticSvmClassifier> svmClassifier = std::dynamic_pointer_cast<TrainableProbabilisticSvmClassifier>(classifier);
		if (!svmClassifier)
			throw invalid_argument("HeadTracking: extended HOG based measurement model (ehog) needs a SVM classifier");
		shared_ptr<ExtendedHogBasedMeasurementModel> model;
		if (config.get<string>("adaptive.measurement.pyramid") == "direct")
			model = make_shared<ExtendedHogBasedMeasurementModel>(svmClassifier);
		else if (config.get<string>("adaptive.measurement.pyramid") == "derived")
			model = make_shared<ExtendedHogBasedMeasurementModel>(svmClassifier, pyramid);
		else
			throw invalid_argument("HeadTracking: invalid pyramid type: " + config.get<string>("adaptive.measurement.pyramid"));
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
			throw invalid_argument("HeadTracking: invalid adaptation type: " + config.get<string>("adaptive.measurement.adaptation"));
		model->setAdaptation(adaptation,
				config.get<double>("adaptive.measurement.adaptationThreshold"),
				config.get<double>("adaptive.measurement.exclusionThreshold"));
		adaptiveMeasurementModel = model;
	} else if (config.get<string>("adaptive.measurement") == "positionDependent") {
		shared_ptr<FeatureExtractor> adaptiveFeatureExtractor = createFeatureExtractor(pyramid, config.get_child("adaptive.measurement.feature"));
		shared_ptr<MeasurementModel> measurementModel;
		if (config.get<string>("adaptive.measurement.filter") == "none") {
			measurementModel = make_shared<SingleClassifierModel>(adaptiveFeatureExtractor, classifier);
		} else {
			// create filter
			shared_ptr<FeatureExtractor> filterFeatureExtractor = createFeatureExtractor(pyramid, config.get_child("adaptive.measurement.filter.feature"));
			filter = RvmClassifier::load(config.get_child("adaptive.measurement.filter"));
			if (config.get<string>("adaptive.measurement.filter") == "before") {
				measurementModel = make_shared<FilteringClassifierModel>(filterFeatureExtractor, filter, adaptiveFeatureExtractor, classifier, FilteringClassifierModel::Behavior::RESET_WEIGHT);
			} else if (config.get<string>("adaptive.measurement.filter") == "after") {
				measurementModel = make_shared<FilteringClassifierModel>(filterFeatureExtractor, filter, adaptiveFeatureExtractor, classifier, FilteringClassifierModel::Behavior::KEEP_WEIGHT);
			} else {
				throw invalid_argument("HeadTracking: invalid filter type: " + config.get<string>("adaptive.measurement.filter"));
			}
		}
		shared_ptr<PositionDependentMeasurementModel> model = make_shared<PositionDependentMeasurementModel>(measurementModel, adaptiveFeatureExtractor, classifier);
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
		adaptiveMeasurementModel = model;
	} else {
		throw invalid_argument("HeadTracking: invalid adaptive measurement model type: " + config.get<string>("adaptive.measurement"));
	}

	// create transition model
	shared_ptr<TransitionModel> transitionModel;
	if (config.get<string>("transition") == "simple") {
		simpleTransitionModel = make_shared<SimpleTransitionModel>(
				config.get<double>("transition.positionDeviation"), config.get<double>("transition.sizeDeviation"));
		transitionModel = simpleTransitionModel;
	} else if (config.get<string>("transition") == "opticalFlow") {
		simpleTransitionModel = make_shared<SimpleTransitionModel>(
				config.get<double>("transition.fallback.positionDeviation"), config.get<double>("transition.fallback.sizeDeviation"));
		opticalFlowTransitionModel = make_shared<OpticalFlowTransitionModel>(
				simpleTransitionModel, config.get<double>("transition.positionDeviation"), config.get<double>("transition.sizeDeviation"));
		transitionModel = opticalFlowTransitionModel;
	} else {
		throw invalid_argument("HeadTracking: invalid transition model type: " + config.get<string>("transition"));
	}

	// create tracker
	adaptiveResamplingSampler = make_shared<ResamplingSampler>(
			config.get<unsigned int>("adaptive.resampling.particleCount"), config.get<double>("adaptive.resampling.randomRate"),
			make_shared<LowVarianceSampling>(), transitionModel,
			config.get<double>("adaptive.resampling.minSize"), config.get<double>("adaptive.resampling.maxSize"));
	gridSampler = make_shared<GridSampler>(config.get<int>("pyramid.patch.minWidth"), config.get<int>("pyramid.patch.maxWidth"),
			1 / pyramid->getIncrementalScaleFactor(), 0.1);
	shared_ptr<StateExtractor> stateExtractor = make_shared<FilteringStateExtractor>(make_shared<WeightedMeanStateExtractor>());
	adaptiveTracker = unique_ptr<AdaptiveCondensationTracker>(new AdaptiveCondensationTracker(
			adaptiveResamplingSampler, adaptiveMeasurementModel, stateExtractor,
			config.get<unsigned int>("adaptive.resampling.particleCount")));
	useAdaptive = true;

	// add validator
	if (config.get<string>("validator") == "rvm") {
		shared_ptr<FeatureExtractor> validatorExtractor = createFeatureExtractor(pyramid, config.get_child("validator.feature"));
		shared_ptr<BinaryClassifier> validatorClassifier = RvmClassifier::load(config.get_child("validator"));
		adaptiveTracker->addValidator(make_shared<ClassificationBasedStateValidator>(
				validatorExtractor, validatorClassifier,
				readValues(config.get<string>("validator.sizes")), readValues(config.get<string>("validator.displacements"))));
	} else if (config.get<string>("validator") != "none") {
		throw invalid_argument("HeadTracking: invalid validator type: " + config.get<string>("validator"));
	}

	if (config.get<string>("initial") == "manual") {
		initialization = Initialization::MANUAL;
	} else if (config.get<string>("initial") == "groundtruth") {
		initialization = Initialization::GROUND_TRUTH;
	} else {
		throw invalid_argument("HeadTracking: invalid initialization type: " + config.get<string>("initial"));
	}
}

void HeadTracking::initGui() {
	drawSamples = false;
	drawFlow = 0;

	cvNamedWindow(videoWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(videoWindowName.c_str(), 50, 50);

	cvNamedWindow(controlWindowName.c_str(), CV_WINDOW_NORMAL);
	cvMoveWindow(controlWindowName.c_str(), 900, 50);

	if (initialTracker) {
		cv::createTrackbar("Adaptive", controlWindowName, NULL, 1, adaptiveChanged, this);
		cv::setTrackbarPos("Adaptive", controlWindowName, useAdaptive ? 1 : 0);
	}

	if (opticalFlowTransitionModel) {
		cv::createTrackbar("Position Deviation * 10", controlWindowName, NULL, 100, positionDeviationChanged, this);
		cv::setTrackbarPos("Position Deviation * 10", controlWindowName, 100 * opticalFlowTransitionModel->getPositionDeviation());

		cv::createTrackbar("Size Deviation * 100", controlWindowName, NULL, 100, sizeDeviationChanged, this);
		cv::setTrackbarPos("Size Deviation * 100", controlWindowName, 100 * opticalFlowTransitionModel->getSizeDeviation());
	} else {
		cv::createTrackbar("Position Deviation * 10", controlWindowName, NULL, 100, positionDeviationChanged, this);
		cv::setTrackbarPos("Position Deviation * 10", controlWindowName, 100 * simpleTransitionModel->getPositionDeviation());

		cv::createTrackbar("Size Deviation * 100", controlWindowName, NULL, 100, sizeDeviationChanged, this);
		cv::setTrackbarPos("Size Deviation * 100", controlWindowName, 100 * simpleTransitionModel->getSizeDeviation());
	}

	if (initialTracker) {
		cv::createTrackbar("Initial Grid/Resampling", controlWindowName, NULL, 1, initialSamplerChanged, this);
		cv::setTrackbarPos("Initial Grid/Resampling", controlWindowName, initialTracker->getSampler() == gridSampler ? 0 : 1);

		cv::createTrackbar("Initial Sample Count", controlWindowName, NULL, 2000, initialSampleCountChanged, this);
		cv::setTrackbarPos("Initial Sample Count", controlWindowName, initialResamplingSampler->getCount());

		cv::createTrackbar("Initial Random Rate", controlWindowName, NULL, 100, initialRandomRateChanged, this);
		cv::setTrackbarPos("Initial Random Rate", controlWindowName, 100 * initialResamplingSampler->getRandomRate());
	}

	cv::createTrackbar("Adaptive Grid/Resampling", controlWindowName, NULL, 1, adaptiveSamplerChanged, this);
	cv::setTrackbarPos("Adaptive Grid/Resampling", controlWindowName, adaptiveTracker->getSampler() == gridSampler ? 0 : 1);

	cv::createTrackbar("Adaptive Sample Count", controlWindowName, NULL, 1000, adaptiveSampleCountChanged, this);
	cv::setTrackbarPos("Adaptive Sample Count", controlWindowName, adaptiveResamplingSampler->getCount());

	cv::createTrackbar("Adaptive Random Rate", controlWindowName, NULL, 100, adaptiveRandomRateChanged, this);
	cv::setTrackbarPos("Adaptive Random Rate", controlWindowName, 100 * adaptiveResamplingSampler->getRandomRate());

	if (filter) {
		cv::createTrackbar("NumFilters", controlWindowName, NULL, filter->getNumFiltersToUse(), numFiltersChanged, this);
		cv::setTrackbarPos("NumFilters", controlWindowName, filter->getNumFiltersToUse());
	}

	cv::createTrackbar("Draw samples", controlWindowName, NULL, 1, drawSamplesChanged, this);
	cv::setTrackbarPos("Draw samples", controlWindowName, drawSamples ? 1 : 0);

	if (opticalFlowTransitionModel) {
		cv::createTrackbar("Draw flow", controlWindowName, NULL, 3, drawFlowChanged, this);
		cv::setTrackbarPos("Draw flow", controlWindowName, drawFlow);
	}
}

void HeadTracking::adaptiveChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->useAdaptive = (state == 1);
	if (state == 0)
		tracking->adaptiveUsable = false;
}

void HeadTracking::positionDeviationChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	if (tracking->opticalFlowTransitionModel)
		tracking->opticalFlowTransitionModel->setPositionDeviation(0.1 * state);
	else
		tracking->simpleTransitionModel->setPositionDeviation(0.1 * state);
}

void HeadTracking::sizeDeviationChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	if (tracking->opticalFlowTransitionModel)
		tracking->opticalFlowTransitionModel->setSizeDeviation(0.01 * state);
	else
		tracking->simpleTransitionModel->setSizeDeviation(0.01 * state);
}

void HeadTracking::initialSamplerChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	if (state == 0)
		tracking->initialTracker->setSampler(tracking->gridSampler);
	else
		tracking->initialTracker->setSampler(tracking->initialResamplingSampler);
}

void HeadTracking::initialSampleCountChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->initialResamplingSampler->setCount(state);
}

void HeadTracking::initialRandomRateChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->initialResamplingSampler->setRandomRate(0.01 * state);
}

void HeadTracking::adaptiveSamplerChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	if (state == 0)
		tracking->adaptiveTracker->setSampler(tracking->gridSampler);
	else
		tracking->adaptiveTracker->setSampler(tracking->adaptiveResamplingSampler);
}

void HeadTracking::adaptiveSampleCountChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->adaptiveResamplingSampler->setCount(state);
}

void HeadTracking::adaptiveRandomRateChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->adaptiveResamplingSampler->setRandomRate(0.01 * state);
}

void HeadTracking::numFiltersChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->filter->setNumFiltersToUse(state);
}

void HeadTracking::drawSamplesChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->drawSamples = (state == 1);
}

void HeadTracking::drawFlowChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->drawFlow = state;
}

void HeadTracking::drawDebug(Mat& image, bool usedAdaptive) {
	cv::Scalar black(0, 0, 0); // blue, green, red
	cv::Scalar red(0, 0, 255); // blue, green, red
	cv::Scalar green(0, 255, 0); // blue, green, red
	if (drawSamples) {
		const std::vector<shared_ptr<Sample>>& samples = usedAdaptive ? adaptiveTracker->getSamples() : initialTracker->getSamples();
		for (const shared_ptr<Sample>& sample : samples) {
			if (!sample->isTarget())
				cv::circle(image, Point(sample->getX(), sample->getY()), 3, black);
		}
		for (const shared_ptr<Sample>& sample : samples) {
			if (sample->isTarget()) {
				cv::Scalar color(0, sample->getWeight() * 255, sample->getWeight() * 255);
				cv::circle(image, Point(sample->getX(), sample->getY()), 3, color);
			}
		}
	}
	if (drawFlow == 1)
		opticalFlowTransitionModel->drawFlow(image, -drawFlow, green, red);
	else if (drawFlow > 1)
		opticalFlowTransitionModel->drawFlow(image, drawFlow - 1, green, red);
}

void HeadTracking::drawCrosshair(Mat& image) {
	if (currentX >= 0 && currentY >= 0 && currentX < image.cols && currentY < image.rows) {
		cv::Scalar gray(127, 127, 127); // blue, green, red
		cv::Scalar black(0, 0, 0); // blue, green, red
		cv::line(image, Point(currentX, 0), Point(currentX, image.rows), gray);
		cv::line(image, Point(0, currentY), Point(image.cols, currentY), gray);
		cv::line(image, Point(currentX, currentY - 5), Point(currentX, currentY + 5), black);
		cv::line(image, Point(currentX - 5, currentY), Point(currentX + 5, currentY), black);
	}
}

void HeadTracking::drawBox(Mat& image) {
	if (storedX >= 0 && storedY >= 0
			&& currentX >= 0 && currentY >= 0 && currentX < image.cols && currentY < image.rows) {
		cv::Scalar color(204, 102, 0); // blue, green, red
		int size = abs(currentY - storedY);
		int x = storedX - size / 2;
		cv::rectangle(image, Point(x, storedY), Point(x + size, currentY), color, 2);
	}
}

void HeadTracking::drawGroundTruth(Mat& image, const LandmarkCollection& landmarks) {
	if (!landmarks.isEmpty())
		landmarks.getLandmark()->draw(image, cv::Scalar(255, 153, 102), 2);
}

void HeadTracking::drawTarget(Mat& image, optional<Rect> target, bool usedAdaptive, bool adapted) {
	const cv::Scalar& color = usedAdaptive ? (adapted ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255)) : cv::Scalar(0, 0, 255);
	if (target)
		cv::rectangle(image, *target, color, 2);
}

void HeadTracking::onMouse(int event, int x, int y, int, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	if (tracking->running && !tracking->adaptiveUsable) {
		if (event == cv::EVENT_MOUSEMOVE) {
			tracking->currentX = x;
			tracking->currentY = y;
			Mat& image = tracking->image;
			tracking->frame.copyTo(image);
			tracking->drawBox(image);
			tracking->drawCrosshair(image);
			imshow(videoWindowName, image);
		} else if (event == cv::EVENT_LBUTTONDOWN) {
			if (tracking->storedX < 0 || tracking->storedY < 0) {
				tracking->storedX = x;
				tracking->storedY = y;
			}
		} else if (event == cv::EVENT_LBUTTONUP) {
			Logger& log = Loggers->getLogger("app");
			int size = abs(tracking->currentY - tracking->storedY);
			Rect position(tracking->storedX - size / 2, std::min(tracking->storedY, tracking->currentY), size, size);

			if (position.width != 0 && position.height != 0) {
				int tries = 0;
				while (tries < 10 && !tracking->adaptiveUsable) {
					tries++;
					tracking->adaptiveUsable = !!tracking->adaptiveTracker->initialize(tracking->frame, position);
				}
				if (tracking->adaptiveUsable) {
					log.info("Initialized adaptive tracking after " + std::to_string(tries) + " tries");
					tracking->storedX = -1;
					tracking->storedY = -1;
					tracking->currentX = -1;
					tracking->currentY = -1;
					Mat& image = tracking->image;
					tracking->frame.copyTo(image);
					tracking->drawTarget(image, optional<Rect>(position), true, true);
					imshow(videoWindowName, image);
				} else {
					log.warn("Could not initialize tracker after " + std::to_string(tries) + " tries (patch too small/big?)");
					std::cerr << "Could not initialize tracker - press 'q' to quit program" << std::endl;
					tracking->stop();
					while ('q' != (char)cv::waitKey(10));
				}
			}
		}
	}
}

void HeadTracking::run() {
	Logger& log = Loggers->getLogger("app");
	running = true;
	paused = false;
	adaptiveUsable = false;

	// manual initialization
	if (initialization == Initialization::MANUAL) {
		storedX = -1;
		storedY = -1;
		currentX = -1;
		currentY = -1;
		cv::setMouseCallback(videoWindowName, onMouse, this);
		while (running && !adaptiveUsable) {
			if (!imageSource->next()) {
				std::cerr << "Could not capture frame - press 'q' to quit program" << std::endl;
				stop();
				while ('q' != (char)cv::waitKey(10));
			} else {
				frame = imageSource->getImage();
				frame.copyTo(image);
				drawBox(image);
				drawCrosshair(image);
				imshow(videoWindowName, image);
				if (imageSink.get() != 0)
					imageSink->add(image);

				int delay = paused ? 0 : 5;
				char c = (char)cv::waitKey(delay);
				if (c == 'p')
					paused = !paused;
				else if (c == 'q')
					stop();
			}
		}
	} else if (initialization == Initialization::GROUND_TRUTH) {
		Logger& log = Loggers->getLogger("app");
		int tries = 0;
		int frameIndex = 0;
		while (running && !adaptiveUsable) {
			if (!imageSource->next()) {
				std::cerr << "Could not capture frame - press 'q' to quit program" << std::endl;
				stop();
				while ('q' != (char)cv::waitKey(10));
			} else {
				frame = imageSource->getImage();
				frame.copyTo(image);
				drawGroundTruth(image, imageSource->getLandmarks());
				if (!imageSource->getLandmarks().isEmpty()) {
					shared_ptr<Landmark> landmark = imageSource->getLandmarks().getLandmark();
					cv::Rect_<float> floatBounds = landmark->getRect();
					if (floatBounds.height > floatBounds.width) {
						int diff = floatBounds.height - floatBounds.width;
						floatBounds.width += diff;
						floatBounds.x -= diff / 2;
					} else if (floatBounds.width > floatBounds.height) {
						int diff = floatBounds.width - floatBounds.height;
						floatBounds.height += diff;
						floatBounds.y -= diff / 2;
					}
					Rect bounds(
							Point(cvRound(floatBounds.tl().x), cvRound(floatBounds.tl().y)),
							Point(cvRound(floatBounds.br().x), cvRound(floatBounds.br().y)));
					if (landmark->isVisible() && bounds.x >= 0 && bounds.y >= 0 && bounds.br().x < image.cols && bounds.br().y < image.rows) {
						tries++;
						adaptiveUsable = !!adaptiveTracker->initialize(frame, bounds);
						drawTarget(image, optional<Rect>(bounds), true, true);
						if (adaptiveUsable) {
							log.info("Initialized adaptive tracking after " + std::to_string(tries) + " tries");
						} else if (tries == 10) {
							log.warn("Could not initialize tracker after " + std::to_string(tries) + " tries (patch too small/big?)");
							std::cerr << "Could not initialize tracker - press 'q' to quit program" << std::endl;
							stop();
							while ('q' != (char)cv::waitKey(10));
						}
					}
				}
				imshow(videoWindowName, image);
				if (imageSink.get() != 0)
					imageSink->add(image);

				int delay = paused ? 0 : 5;
				char c = (char)cv::waitKey(delay);
				if (c == 'p')
					paused = !paused;
				else if (c == 'q')
					stop();
				frameIndex++;
			}
		}
	}

	duration<double> allIterationTime;
	duration<double> allCondensationTime;
	int frames = 0;

	// adaptive tracking
	while (running) {
		steady_clock::time_point frameStart = steady_clock::now();

		if (!imageSource->next()) {
			std::cerr << "Could not capture frame - press 'q' to quit program" << std::endl;
			stop();
			while ('q' != (char)cv::waitKey(10));
		} else {
			frames++;
			frame = imageSource->getImage();
			steady_clock::time_point condensationStart = steady_clock::now();
			bool usedAdaptive = false;
			bool adapted = false;
			optional<Rect> position = adaptiveTracker->process(frame);
			usedAdaptive = true;
			adapted = adaptiveTracker->hasAdapted();
			steady_clock::time_point condensationEnd = steady_clock::now();
			frame.copyTo(image);
			drawDebug(image, usedAdaptive);
			drawGroundTruth(image, imageSource->getLandmarks());
			drawTarget(image, position, usedAdaptive, adapted);
			imshow(videoWindowName, image);
			if (imageSink.get() != 0)
				imageSink->add(image);
			steady_clock::time_point frameEnd = steady_clock::now();

			milliseconds iterationTime = duration_cast<milliseconds>(frameEnd - frameStart);
			milliseconds condensationTime = duration_cast<milliseconds>(condensationEnd - condensationStart);
			allIterationTime += iterationTime;
			allCondensationTime += condensationTime;
			float iterationFps = frames / allIterationTime.count();
			float condensationFps = frames / allCondensationTime.count();

			ostringstream text;
			text.precision(2);
			text << frames << " frame: " << iterationTime.count() << " ms (" << iterationFps << " fps);"
					<< " condensation: " << condensationTime.count() << " ms (" << condensationFps << " fps)";
			log.info(text.str());

			int delay = paused ? 0 : 5;
			char c = (char)cv::waitKey(delay);
			if (c == 'p')
				paused = !paused;
			else if (c == 'q')
				stop();
		}
	}
}

void HeadTracking::stop() {
	running = false;
}

int main(int argc, char *argv[]) {
	int verboseLevelText;
	int verboseLevelImages;
	int deviceId, kinectId;
	string filename, directory, groundTruthFilename;
	bool useCamera = false, useKinect = false, useFile = false, useDirectory = false, useGroundTruth = false, bobot = false;
	string configFile;
	string outputFile;
	int outputFps = -1;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "Produce help message")
			("verbose-text,v", po::value<int>(&verboseLevelText)->implicit_value(2)->default_value(0,"minimal text output"), "Enable text-verbosity (optionally specify level)")
			("verbose-images,w", po::value<int>(&verboseLevelImages)->implicit_value(2)->default_value(0,"minimal image output"), "Enable image-verbosity (optionally specify level)")
			("filename,f", po::value< string >(&filename), "A filename of a video to run the tracking")
			("directory,i", po::value< string >(&directory), "Use a directory as input")
			("device,d", po::value<int>(&deviceId)->implicit_value(0), "A camera device ID for use with the OpenCV camera driver")
			("kinect,k", po::value<int>(&kinectId)->implicit_value(0), "Windows only: Use a Kinect as camera. Optionally specify a device ID.")
			("ground-truth,g", po::value<string>(&groundTruthFilename), "Name of a file containing ground truth information in BoBoT format")
			("bobot,b", "Flag for indicating BoBoT format on the ground truth file")
			("config,c", po::value< string >(&configFile)->default_value("default.cfg","default.cfg"), "The filename to the config file.")
			("output,o", po::value< string >(&outputFile)->default_value("","none"), "Filename to a video file for storing the image data.")
			("output-fps,r", po::value<int>(&outputFps)->default_value(-1), "The framerate of the output video.")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
			std::cout << "Usage: adaptiveTrackingApp [options]" << std::endl;
			std::cout << desc;
			return 0;
		}
		if (vm.count("filename"))
			useFile = true;
		if (vm.count("directory"))
			useDirectory = true;
		if (vm.count("device"))
			useCamera = true;
		if (vm.count("kinect"))
			useKinect = true;
		if (vm.count("ground-truth"))
			useGroundTruth = true;
		if (vm.count("bobot"))
			bobot = true;
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return -1;
	}

	int inputsSpecified = 0;
	if (useCamera)
		inputsSpecified++;
	if (useKinect)
		inputsSpecified++;
	if (useFile)
		inputsSpecified++;
	if (useDirectory)
		inputsSpecified++;
	if (inputsSpecified != 1) {
		std::cout << "Usage: Please specify a camera, Kinect, file or directory (and only one of them) to run the program. Use -h for help." << std::endl;
		return -1;
	}

	Loggers->getLogger("app").addAppender(make_shared<ConsoleAppender>(LogLevel::Info));

	shared_ptr<ImageSource> imageSource;
	if (useCamera)
		imageSource.reset(new CameraImageSource(deviceId));
	else if (useKinect)
		imageSource.reset(new KinectImageSource(kinectId));
	else if (useFile)
		imageSource.reset(new VideoImageSource(filename));
	else if (useDirectory)
		imageSource.reset(new DirectoryImageSource(directory));
	shared_ptr<LandmarkSource> landmarkSource;
	if (useGroundTruth && bobot)
		landmarkSource.reset(new BobotLandmarkSource(groundTruthFilename, imageSource));
	else if (useGroundTruth)
		landmarkSource.reset(new SingleLandmarkSource(groundTruthFilename));
	else
		landmarkSource.reset(new EmptyLandmarkSource());
	unique_ptr<LabeledImageSource> labeledImageSource(new OrderedLabeledImageSource(imageSource, landmarkSource));

	unique_ptr<ImageSink> imageSink;
	if (outputFile != "") {
		if (outputFps < 0) {
			std::cout << "Usage: You have to specify the framerate of the output video file by using option -r. Use -h for help." << std::endl;
			return -1;
		}
		imageSink.reset(new VideoImageSink(outputFile, outputFps));
	}

	ptree config;
	read_info(configFile, config);
	if (useGroundTruth)
		config.put("tracking.initial", "groundtruth");
	try {
		unique_ptr<HeadTracking> tracker(new HeadTracking(move(labeledImageSource), move(imageSink), config.get_child("tracking")));
		tracker->run();
	} catch (std::exception& exc) {
		Loggers->getLogger("app").error(string("A wild exception appeared: ") + exc.what());
		throw;
	}
	return 0;
}
