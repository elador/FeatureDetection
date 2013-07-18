/*
 * HeadTracking.cpp
 *
 *  Created on: 14.05.2013
 *      Author: poschmann
 */

#include "HeadTracking.hpp"
#include "PositionDependentMeasurementModel2.hpp"
#include "logging/LoggerFactory.hpp"
#include "logging/Logger.hpp"
#include "logging/ConsoleAppender.hpp"
#include "imageio/OrderedLandmarkSource.hpp"
#include "imageio/BobotLandmarkSource.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
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
#include "imageprocessing/GradientHistogramFilter.hpp"
#include "imageprocessing/ConversionFilter.hpp"
#include "imageprocessing/IntensityNormNormalizationFilter.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/DirectImageFeatureExtractor.hpp"
#include "imageprocessing/FilteringFeatureExtractor.hpp"
#include "imageprocessing/FilteringPyramidFeatureExtractor.hpp"
#include "imageprocessing/PatchResizingFeatureExtractor.hpp"
#include "imageprocessing/SpatialHistogramFeatureExtractor.hpp"
#include "imageprocessing/SpatialPyramidHistogramFeatureExtractor.hpp"
#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/RbfKernel.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/HikKernel.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/RvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/FrameBasedTrainableSvmClassifier.hpp"
#include "classification/TrainableProbabilisticTwoStageClassifier.hpp"
#include "classification/FixedSizeTrainableSvmClassifier.hpp"
#include "classification/FixedTrainableProbabilisticSvmClassifier.hpp"
#include "condensation/ResamplingSampler.hpp"
#include "condensation/GridSampler.hpp"
#include "condensation/LowVarianceSampling.hpp"
#include "condensation/SimpleTransitionModel.hpp"
#include "condensation/OpticalFlowTransitionModel.hpp"
#include "condensation/WvmSvmModel.hpp"
#include "condensation/SelfLearningMeasurementModel.hpp"
#include "condensation/FilteringPositionExtractor.hpp"
#include "condensation/WeightedMeanPositionExtractor.hpp"
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
using cv::Point;
using boost::property_tree::info_parser::read_info;
using boost::lexical_cast;
using std::milli;
using std::move;
using std::ostringstream;
using std::istringstream;

namespace po = boost::program_options;

const string HeadTracking::videoWindowName = "Image";
const string HeadTracking::controlWindowName = "Controls";

HeadTracking::HeadTracking(unique_ptr<LabeledImageSource> imageSource, unique_ptr<ImageSink> imageSink, ptree& config) :
		imageSource(move(imageSource)), imageSink(move(imageSink)) {
	initTracking(config);
	initGui();
}

HeadTracking::~HeadTracking() {}

shared_ptr<FeatureExtractor> HeadTracking::createFeatureExtractor(
		shared_ptr<DirectPyramidFeatureExtractor> patchExtractor, ptree& config) {
	float scaleFactor = config.get<float>("scale");
	if (config.get_value<string>() == "histeq") {
		shared_ptr<FilteringFeatureExtractor> featureExtractor = make_shared<FilteringFeatureExtractor>(patchExtractor);
		featureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
		return wrapFeatureExtractor(featureExtractor, scaleFactor);
	} else if (config.get_value<string>() == "whi") {
		shared_ptr<FilteringFeatureExtractor> featureExtractor = make_shared<FilteringFeatureExtractor>(patchExtractor);
		featureExtractor->addPatchFilter(make_shared<WhiteningFilter>());
		featureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
		featureExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0 / 127.5, -1.0));
		featureExtractor->addPatchFilter(make_shared<IntensityNormNormalizationFilter>());
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
		patchExtractor->addLayerFilter(make_shared<GradientFilter>(config.get<int>("gradientKernel"), config.get<int>("blurKernel")));
		patchExtractor->addLayerFilter(make_shared<GradientHistogramFilter>(config.get<int>("bins"), config.get<bool>("signed"), config.get<float>("offset")));
		return wrapFeatureExtractor(createHistogramFeatureExtractor(patchExtractor,
				config.get<int>("bins"), config.get_child("histogram")), scaleFactor);
	} else if (config.get_value<string>() == "lbp") {
		shared_ptr<LbpFilter> lbpFilter = createLbpFilter(config.get<string>("type"));
		patchExtractor->addLayerFilter(lbpFilter);
		return wrapFeatureExtractor(createHistogramFeatureExtractor(patchExtractor,
				lbpFilter->getBinCount(), config.get_child("histogram")), scaleFactor);
	} else if (config.get_value<string>() == "glbp") {
		shared_ptr<LbpFilter> lbpFilter = createLbpFilter(config.get<string>("type"));
		patchExtractor->addLayerFilter(make_shared<GradientFilter>(config.get<int>("gradientKernel"), config.get<int>("blurKernel")));
		patchExtractor->addLayerFilter(make_shared<GradientMagnitudeFilter>());
		patchExtractor->addLayerFilter(lbpFilter);
		return wrapFeatureExtractor(createHistogramFeatureExtractor(patchExtractor,
				lbpFilter->getBinCount(), config.get_child("histogram")), scaleFactor);
	} else {
		throw invalid_argument("AdaptiveTracking: invalid feature type: " + config.get<string>("feature"));
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
		throw invalid_argument("AdaptiveTracking: invalid LBP type: " + lbpType);
	return make_shared<LbpFilter>(type);
}

shared_ptr<FeatureExtractor> HeadTracking::createHistogramFeatureExtractor(
		shared_ptr<FeatureExtractor> patchExtractor, unsigned int bins, ptree& config) {
	HistogramFeatureExtractor::Normalization normalization;
	if (config.get<string>("normalization") == "none")
		normalization = HistogramFeatureExtractor::Normalization::NONE;
	else if (config.get<string>("normalization") == "l2norm")
		normalization = HistogramFeatureExtractor::Normalization::L2NORM;
	else if (config.get<string>("normalization") == "l2hys")
		normalization = HistogramFeatureExtractor::Normalization::L2HYS;
	else if (config.get<string>("normalization") == "l1norm")
		normalization = HistogramFeatureExtractor::Normalization::L1NORM;
	else if (config.get<string>("normalization") == "l1sqrt")
		normalization = HistogramFeatureExtractor::Normalization::L1SQRT;
	else
		throw invalid_argument("AdaptiveTracking: invalid normalization method: " + config.get<string>("normalization"));
	if (config.get_value<string>() == "spatial")
		return make_shared<SpatialHistogramFeatureExtractor>(patchExtractor, bins,
				config.get<int>("cellSize"), config.get<int>("blockSize"), config.get<bool>("combine"), normalization);
	else 	if (config.get_value<string>() == "pyramid")
		return make_shared<SpatialPyramidHistogramFeatureExtractor>(patchExtractor, bins, config.get<int>("level"), normalization);
	else
		throw invalid_argument("AdaptiveTracking: invalid histogram type: " + config.get_value<string>());
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
		return make_shared<HikKernel>();
	} else if (config.get_value<string>() == "linear") {
		return make_shared<LinearKernel>();
	} else {
		throw invalid_argument("AdaptiveTracking: invalid kernel type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableSvmClassifier> HeadTracking::createTrainableSvm(shared_ptr<Kernel> kernel, ptree& config) {
	shared_ptr<TrainableSvmClassifier> trainableSvm;
	if (config.get_value<string>() == "fixedsize") {
		trainableSvm = make_shared<FixedSizeTrainableSvmClassifier>(kernel, config.get<double>("constraintsViolationCosts"),
				config.get<unsigned int>("positiveExamples"), config.get<unsigned int>("negativeExamples"), config.get<unsigned int>("minPositiveExamples"));
	} else if (config.get_value<string>() == "framebased") {
		trainableSvm = make_shared<FrameBasedTrainableSvmClassifier>(kernel, config.get<double>("constraintsViolationCosts"),
				config.get<int>("frameLength"), config.get<float>("minAvgSamples"));
	} else {
		throw invalid_argument("AdaptiveTracking: invalid training type: " + config.get_value<string>());
	}
	optional<ptree&> negativesConfig = config.get_child_optional("staticNegatives");
	if (negativesConfig && negativesConfig->get_value<bool>()) {
		trainableSvm->loadStaticNegatives(negativesConfig->get<string>("filename"),
				negativesConfig->get<int>("amount"), negativesConfig->get<double>("scale"));
	}
	return trainableSvm;
}

shared_ptr<TrainableProbabilisticClassifier> HeadTracking::createClassifier(
		shared_ptr<TrainableSvmClassifier> trainableSvm, ptree& config) {
	if (config.get_value<string>() == "default")
		return make_shared<TrainableProbabilisticSvmClassifier>(trainableSvm);
	else if (config.get_value<string>() == "fixed")
		return make_shared<FixedTrainableProbabilisticSvmClassifier>(trainableSvm);
	else
		throw invalid_argument("AdaptiveTracking: invalid classifier type: " + config.get_value<string>());
}

void HeadTracking::initTracking(ptree& config) {
	// create patch extractor
	patchExtractor = make_shared<DirectPyramidFeatureExtractor>(
			config.get<int>("pyramid.patch.width"), config.get<int>("pyramid.patch.height"),
			config.get<int>("pyramid.patch.minWidth"), config.get<int>("pyramid.patch.maxWidth"),
			config.get<double>("pyramid.scaleFactor"));
	patchExtractor->addImageFilter(make_shared<GrayscaleFilter>());

	// create filter
	shared_ptr<DirectPyramidFeatureExtractor> filterFeatureExtractor;
	if (config.get<string>("filter.feature") == "grayscale") {
		filterFeatureExtractor = make_shared<DirectPyramidFeatureExtractor>(31, 31, 31, 480, 0.9); // w, h, min, max, scale
		filterFeatureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
		filterFeatureExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0 / 255.0));
	} else if (config.get<string>("filter.feature") == "whi") {
		filterFeatureExtractor = make_shared<DirectPyramidFeatureExtractor>(31, 31, 31, 480, 0.9); // w, h, min, max, scale
		filterFeatureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
		filterFeatureExtractor->addLayerFilter(make_shared<WhiteningFilter>());
		filterFeatureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
		filterFeatureExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0 / 127.5, -1.0));
		filterFeatureExtractor->addPatchFilter(make_shared<IntensityNormNormalizationFilter>());
	}
	filter = ProbabilisticRvmClassifier::load(config.get_child("filter"));

	// create adaptive measurement model
	adaptiveFeatureExtractor = createFeatureExtractor(patchExtractor, config.get_child("adaptive.feature"));
	shared_ptr<Kernel> kernel = createKernel(config.get_child("adaptive.measurement.classifier.kernel"));
	shared_ptr<TrainableSvmClassifier> trainableSvm = createTrainableSvm(kernel, config.get_child("adaptive.measurement.classifier.training"));
	shared_ptr<TrainableProbabilisticClassifier> classifier = createClassifier(trainableSvm, config.get_child("adaptive.measurement.classifier.probabilistic"));
	adaptiveMeasurementModel.reset(new PositionDependentMeasurementModel2(
			filterFeatureExtractor, filter, adaptiveFeatureExtractor, classifier,
			config.get<int>("adaptive.measurement.startFrameCount"), config.get<int>("adaptive.measurement.stopFrameCount"),
			config.get<float>("adaptive.measurement.targetThreshold"), config.get<float>("adaptive.measurement.confidenceThreshold"),
			config.get<float>("adaptive.measurement.positiveOffsetFactor"), config.get<float>("adaptive.measurement.negativeOffsetFactor"),
			config.get<int>("adaptive.measurement.sampleNegativesAroundTarget"), config.get<bool>("adaptive.measurement.sampleFalsePositives"),
			config.get<unsigned int>("adaptive.measurement.randomNegatives"), config.get<bool>("adaptive.measurement.exploitSymmetry")));

	// create transition model
	shared_ptr<TransitionModel> transitionModel;
	if (config.get<string>("transition") == "simple") {
		simpleTransitionModel = make_shared<SimpleTransitionModel>(
				config.get<double>("transition.positionScatter"), config.get<double>("transition.velocityScatter"));
		transitionModel = simpleTransitionModel;
	} else if (config.get<string>("transition") == "opticalFlow") {
		simpleTransitionModel = make_shared<SimpleTransitionModel>(
				config.get<double>("transition.fallback.positionScatter"), config.get<double>("transition.fallback.velocityScatter"));
		opticalFlowTransitionModel = make_shared<OpticalFlowTransitionModel>(simpleTransitionModel, config.get<double>("transition.scatter"));
		transitionModel = opticalFlowTransitionModel;
	} else {
		throw invalid_argument("AdaptiveTracking: invalid transition model type: " + config.get<string>("transition"));
	}

	// create tracker
	adaptiveResamplingSampler = make_shared<ResamplingSampler>(
			config.get<unsigned int>("adaptive.resampling.particleCount"), config.get<double>("adaptive.resampling.randomRate"),
			make_shared<LowVarianceSampling>(), transitionModel,
			config.get<double>("adaptive.resampling.minSize"), config.get<double>("adaptive.resampling.maxSize"));
	gridSampler = make_shared<GridSampler>(config.get<int>("pyramid.patch.minWidth"), config.get<int>("pyramid.patch.maxWidth"),
			1 / config.get<double>("pyramid.scaleFactor"), 0.1);
	shared_ptr<PositionExtractor> positionExtractor
			= make_shared<FilteringPositionExtractor>(make_shared<WeightedMeanPositionExtractor>());
	adaptiveTracker = unique_ptr<AdaptiveCondensationTracker>(new AdaptiveCondensationTracker(
			adaptiveResamplingSampler, adaptiveMeasurementModel, positionExtractor,
			config.get<unsigned int>("adaptive.resampling.particleCount")));
	useAdaptive = true;

	initialization = Initialization::MANUAL;
}

void HeadTracking::initGui() {
	drawSamples = true;
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
		cv::createTrackbar("Scatter * 100", controlWindowName, NULL, 100, scatterChanged, this);
		cv::setTrackbarPos("Scatter * 100", controlWindowName, 100 * opticalFlowTransitionModel->getScatter());
	}

	if (simpleTransitionModel) {
		cv::createTrackbar("Position Scatter * 100", controlWindowName, NULL, 100, positionScatterChanged, this);
		cv::setTrackbarPos("Position Scatter * 100", controlWindowName, 100 * simpleTransitionModel->getPositionScatter());

		cv::createTrackbar("Velocity Scatter * 100", controlWindowName, NULL, 100, velocityScatterChanged, this);
		cv::setTrackbarPos("Velocity Scatter * 100", controlWindowName, 100 * simpleTransitionModel->getVelocityScatter());
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
		cv::createTrackbar("NumFilters", controlWindowName, NULL, filter->getRvm()->getNumFiltersToUse(), numFiltersChanged, this);
		cv::setTrackbarPos("NumFilters", controlWindowName, filter->getRvm()->getNumFiltersToUse());
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

void HeadTracking::scatterChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->opticalFlowTransitionModel->setScatter(0.01 * state);
}

void HeadTracking::positionScatterChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->simpleTransitionModel->setPositionScatter(0.01 * state);
}

void HeadTracking::velocityScatterChanged(int state, void* userdata) {
	HeadTracking *tracking = (HeadTracking*)userdata;
	tracking->simpleTransitionModel->setVelocityScatter(0.01 * state);
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
	tracking->filter->getRvm()->setNumFiltersToUse(state);
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
		const std::vector<Sample>& samples = usedAdaptive ? adaptiveTracker->getSamples() : initialTracker->getSamples();
		for (auto sit = samples.cbegin(); sit < samples.cend(); ++sit) {
			if (!sit->isObject())
				cv::circle(image, Point(sit->getX(), sit->getY()), 3, black);
		}
		for (auto sit = samples.cbegin(); sit < samples.cend(); ++sit) {
			if (sit->isObject()) {
				cv::Scalar color(0, sit->getWeight() * 255, sit->getWeight() * 255);
				cv::circle(image, Point(sit->getX(), sit->getY()), 3, color);
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

void HeadTracking::drawTarget(Mat& image, optional<Rect> target, optional<Sample> state, bool usedAdaptive, bool adapted) {
	const cv::Scalar& color = usedAdaptive ? (adapted ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255)) : cv::Scalar(0, 0, 255);
	if (target) {
		cv::rectangle(image, *target, color, 2);
		if (state)
			cv::line(image,
					cv::Point(state->getX() - state->getVx(), state->getY() - state->getVy()),
					cv::Point(state->getX(), state->getY()), color);
	}
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
					tracking->adaptiveUsable = tracking->adaptiveTracker->initialize(tracking->frame, position);
				}
				if (tracking->adaptiveUsable) {
					log.info("Initialized adaptive tracking after " + lexical_cast<string>(tries) + " tries");
					tracking->storedX = -1;
					tracking->storedY = -1;
					tracking->currentX = -1;
					tracking->currentY = -1;
					Mat& image = tracking->image;
					tracking->frame.copyTo(image);
					tracking->drawTarget(image, optional<Rect>(position), optional<Sample>(), true, true);
					imshow(videoWindowName, image);
				} else {
					log.warn("Could not initialize tracker after " + lexical_cast<string>(tries) + " tries (patch too small/big?)");
					std::cerr << "Could not initialize tracker - press 'q' to quit program" << std::endl;
					tracking->stop();
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
			boost::optional<Rect> position;
			boost::optional<Sample> state;
			position = adaptiveTracker->process(frame);
			state = adaptiveTracker->getState();
			usedAdaptive = true;
			adapted = adaptiveTracker->hasAdapted();
			steady_clock::time_point condensationEnd = steady_clock::now();
			frame.copyTo(image);
			drawDebug(image, usedAdaptive);
			drawTarget(image, position, state, usedAdaptive, adapted);
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
	bool useCamera = false, useKinect = false, useFile = false, useDirectory = false, useGroundTruth = false;
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

	Loggers->getLogger("app").addAppender(make_shared<ConsoleAppender>(loglevel::INFO));

	shared_ptr<ImageSource> imageSource;
	if (useCamera)
		imageSource.reset(new VideoImageSource(deviceId));
	else if (useKinect)
		imageSource.reset(new KinectImageSource(kinectId));
	else if (useFile)
		imageSource.reset(new VideoImageSource(filename));
	else if (useDirectory)
		imageSource.reset(new DirectoryImageSource(directory));
	shared_ptr<OrderedLandmarkSource> landmarkSource;
	if (useGroundTruth)
		landmarkSource.reset(new BobotLandmarkSource(imageSource, groundTruthFilename));
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
