/*
 * DetectorTrainingApp.cpp
 *
 *  Created on: 23.10.2015
 *      Author: poschmann
 */

#include "stacktrace.hpp"
#include "boost/filesystem.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/property_tree/ptree.hpp"
#include "DetectorTester.hpp"
#include "DetectorTrainer.hpp"
#include "DilatedLabeledImageSource.hpp"
#include "LabeledImage.hpp"
#include "classification/SvmClassifier.hpp"
#include "imageio/DlibImageSource.hpp"
#include "imageprocessing/ChainedFilter.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/extraction/AggregatedFeaturesExtractor.hpp"
#include "imageprocessing/filtering/AggregationFilter.hpp"
#include "imageprocessing/filtering/FhogAggregationFilter.hpp"
#include "imageprocessing/filtering/FhogFilter.hpp"
#include "imageprocessing/filtering/FpdwFeaturesFilter.hpp"
#include "imageprocessing/filtering/GradientFilter.hpp"
#include "imageprocessing/filtering/GradientHistogramFilter.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

using boost::filesystem::path;
using boost::filesystem::exists;
using boost::filesystem::create_directory;
using boost::property_tree::ptree;
using boost::property_tree::read_info;
using boost::property_tree::write_info;
using cv::Mat;
using cv::Rect;
using cv::Size;
using classification::SvmClassifier;
using detection::AggregatedFeaturesDetector;
using detection::NonMaximumSuppression;
using imageio::DlibImageSource;
using imageio::LabeledImageSource;
using imageio::Landmark;
using imageprocessing::ChainedFilter;
using imageprocessing::GrayscaleFilter;
using imageprocessing::ImageFilter;
using imageprocessing::ImagePyramid;
using imageprocessing::extraction::AggregatedFeaturesExtractor;
using imageprocessing::filtering::AggregationFilter;
using imageprocessing::filtering::FhogAggregationFilter;
using imageprocessing::filtering::FhogFilter;
using imageprocessing::filtering::FpdwFeaturesFilter;
using imageprocessing::filtering::GradientFilter;
using imageprocessing::filtering::GradientHistogramFilter;
using std::cerr;
using std::chrono::duration_cast;
using std::chrono::seconds;
using std::chrono::steady_clock;
using std::cout;
using std::endl;
using std::invalid_argument;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;


enum class TaskType { TRAIN, TEST, SHOW };
enum class FeatureType { FHOG, FAST_FHOG, GRADHIST, FPDW };

/**
 * Parameters of the detection.
 */
struct DetectionParams {
	cv::Size minWindowSizeInPixels; ///< Smallest window size that is detected in pixels.
	int octaveLayerCount; ///< Number of image pyramid layers per octave.
	bool approximatePyramid; ///< Flag that indicates whether to approximate all but one image pyramid layer per octave.
	double nmsOverlapThreshold; ///< Maximum allowed overlap between two detections.
};

vector<LabeledImage> getLabeledImages(shared_ptr<LabeledImageSource> source, FeatureParams featureParams) {
	vector<LabeledImage> images;
	while (source->next())
		images.emplace_back(source->getImage(), source->getLandmarks().getLandmarks());
	double width = featureParams.windowSizeInCells.width / featureParams.widthScaleFactor;
	double height = featureParams.windowSizeInCells.height / featureParams.heightScaleFactor;
	double aspectRatio = width / height;
	for (LabeledImage& image : images)
		image.adjustSizes(aspectRatio);
	return images;
}

vector<vector<LabeledImage>> getSubsets(const vector<LabeledImage>& labeledImages, int setCount) {
	vector<vector<LabeledImage>> subsets(setCount);
	for (int i = 0; i < labeledImages.size(); ++i)
		subsets[i % setCount].push_back(labeledImages[i]);
	return subsets;
}

vector<LabeledImage> getTrainingSet(const vector<vector<LabeledImage>>& subsets, int testSetIndex) {
	vector<LabeledImage> trainingSet;
	for (int i = 0; i < subsets.size(); ++i) {
		if (i != testSetIndex) {
			std::copy(subsets[i].begin(), subsets[i].end(), std::back_inserter(trainingSet));
		}
	}
	return trainingSet;
}

shared_ptr<ImageFilter> createFhogFilter(FeatureParams featureParams, bool fast) {
	if (fast) {
		return make_shared<FhogFilter>(featureParams.cellSizeInPixels, 9, false, true, 0.2f);
//		return make_shared<FhogFilter>(featureParams.cellSizeInPixels, 8, false, true, 0.225f);
//		return make_shared<FhogFilter>(featureParams.cellSizeInPixels, 7, false, true, 0.257f);
//		return make_shared<FhogFilter>(featureParams.cellSizeInPixels, 6, false, true, 0.3f);
//		return make_shared<FhogFilter>(featureParams.cellSizeInPixels, 5, false, true, 0.36f);
	} else {
		auto gradientFilter = make_shared<GradientFilter>(1);
		auto gradientHistogramFilter = make_shared<GradientHistogramFilter>(18, false, false, true, false, 0);
		auto aggregationFilter = make_shared<FhogAggregationFilter>(featureParams.cellSizeInPixels, true, 0.2f);
		return make_shared<ChainedFilter>(gradientFilter, gradientHistogramFilter, aggregationFilter);
	}
}

shared_ptr<ImageFilter> createGradientFeaturesFilter(FeatureParams featureParams) {
	auto gradientFilter = make_shared<GradientFilter>(1);
	int normalizationRadius = featureParams.cellSizeInPixels;
//	auto gradientHistogramFilter = GradientHistogramFilter::full(12, true, normalizationRadius); // 12 + 1
//	auto gradientHistogramFilter = GradientHistogramFilter::half(6, true, normalizationRadius); // 6 + 1
	auto gradientHistogramFilter = GradientHistogramFilter::both(6, true, normalizationRadius); // 12 + 6 + 1
	auto aggregationFilter = make_shared<AggregationFilter>(featureParams.cellSizeInPixels, true, false);
	return make_shared<ChainedFilter>(gradientFilter, gradientHistogramFilter, aggregationFilter);
}

shared_ptr<ImageFilter> createFdpwFilter(FeatureParams featureParams) {
	auto fpdwFeatures = make_shared<FpdwFeaturesFilter>(true, false, featureParams.cellSizeInPixels, 0.01);
	auto aggregation = make_shared<AggregationFilter>(featureParams.cellSizeInPixels, true, false);
	return make_shared<ChainedFilter>(fpdwFeatures, aggregation);
}

shared_ptr<AggregatedFeaturesDetector> createDetector(const shared_ptr<SvmClassifier>& svm,
		const shared_ptr<ImageFilter>& imageFilter, const shared_ptr<ImageFilter>& layerFilter,
		FeatureParams featureParams, DetectionParams detectionParams) {
	shared_ptr<NonMaximumSuppression> nms = make_shared<NonMaximumSuppression>(
			detectionParams.nmsOverlapThreshold, NonMaximumSuppression::MaximumType::MAX_SCORE);
	if (imageFilter)
		return make_shared<AggregatedFeaturesDetector>(imageFilter, layerFilter, featureParams.cellSizeInPixels,
				featureParams.windowSizeInCells, detectionParams.octaveLayerCount, svm, nms,
				featureParams.widthScaleFactorInv(), featureParams.heightScaleFactorInv(), detectionParams.minWindowSizeInPixels.width);
	else
		return make_shared<AggregatedFeaturesDetector>(layerFilter, featureParams.cellSizeInPixels,
				featureParams.windowSizeInCells, detectionParams.octaveLayerCount, svm, nms,
				featureParams.widthScaleFactorInv(), featureParams.heightScaleFactorInv(), detectionParams.minWindowSizeInPixels.width);
}

shared_ptr<AggregatedFeaturesDetector> createApproximateDetector(const shared_ptr<SvmClassifier>& svm,
		const shared_ptr<ImageFilter>& imageFilter, const shared_ptr<ImageFilter>& layerFilter,
		FeatureParams featureParams, DetectionParams detectionParams, vector<double> lambdas) {
	auto featurePyramid = ImagePyramid::createApproximated(detectionParams.octaveLayerCount, 0.5, 1.0, lambdas);
	if (imageFilter)
		featurePyramid->addImageFilter(imageFilter);
	featurePyramid->addLayerFilter(layerFilter);
	auto extractor = make_shared<AggregatedFeaturesExtractor>(featurePyramid, featureParams.windowSizeInCells,
			featureParams.cellSizeInPixels, true, detectionParams.minWindowSizeInPixels.width);
	shared_ptr<NonMaximumSuppression> nms = make_shared<NonMaximumSuppression>(
			detectionParams.nmsOverlapThreshold, NonMaximumSuppression::MaximumType::MAX_SCORE);
	return make_shared<AggregatedFeaturesDetector>(extractor, svm, nms,
			featureParams.widthScaleFactorInv(), featureParams.heightScaleFactorInv());
}

shared_ptr<AggregatedFeaturesDetector> loadDetector(const string& filename, FeatureType featureType,
		FeatureParams featureParams, DetectionParams detectionParams, float threshold = 0) {
	std::ifstream stream(filename);
	shared_ptr<SvmClassifier> svm = SvmClassifier::load(stream);
	svm->setThreshold(threshold);
	stream.close();

	shared_ptr<ImageFilter> imageFilter;
	shared_ptr<ImageFilter> layerFilter;
	vector<double> lambdas;
	if (featureType == FeatureType::FHOG) {
		imageFilter = make_shared<GrayscaleFilter>();
		layerFilter = createFhogFilter(featureParams, false);
		// TODO lambdas
	} else if (featureType == FeatureType::FAST_FHOG) {
		imageFilter = make_shared<GrayscaleFilter>();
		layerFilter = createFhogFilter(featureParams, true);
		// TODO lambdas
	} else if (featureType == FeatureType::GRADHIST) {
		imageFilter = make_shared<GrayscaleFilter>();
		layerFilter = createGradientFeaturesFilter(featureParams);
		// TODO lambdas
	} else if (featureType == FeatureType::FPDW) {
		layerFilter = createFdpwFilter(featureParams);
		// TODO lambdas
	} else {
		throw invalid_argument("unknown feature type");
	}

	if (detectionParams.approximatePyramid)
		return createApproximateDetector(svm, imageFilter, layerFilter, featureParams, detectionParams, lambdas);
	return createDetector(svm, imageFilter, layerFilter, featureParams, detectionParams);
}

void setFeatures(DetectorTrainer& detectorTrainer, FeatureType featureType, FeatureParams featureParams) {
	if (featureType == FeatureType::FHOG)
		detectorTrainer.setFeatures(featureParams, createFhogFilter(featureParams, false), make_shared<GrayscaleFilter>());
	else if (featureType == FeatureType::FAST_FHOG)
		detectorTrainer.setFeatures(featureParams, createFhogFilter(featureParams, true), make_shared<GrayscaleFilter>());
	else if (featureType == FeatureType::GRADHIST)
		detectorTrainer.setFeatures(featureParams, createGradientFeaturesFilter(featureParams), make_shared<GrayscaleFilter>());
	else if (featureType == FeatureType::FPDW)
		detectorTrainer.setFeatures(featureParams, createFdpwFilter(featureParams));
	else
		throw invalid_argument("unknown feature type");
}

bool showDetections(const DetectorTester& tester, AggregatedFeaturesDetector& detector, const vector<LabeledImage>& images) {
	Mat output;
	cv::Scalar correctDetectionColor(0, 255, 0);
	cv::Scalar wrongDetectionColor(0, 0, 255);
	cv::Scalar ignoredDetectionColor(255, 204, 0);
	cv::Scalar missedDetectionColor(0, 153, 255);
	int thickness = 2;
	for (const LabeledImage& image : images) {
		DetectionResult result = tester.detect(detector, image.image, image.landmarks);
		image.image.copyTo(output);
		for (const Rect& target : result.correctDetections)
			cv::rectangle(output, target, correctDetectionColor, thickness);
		for (const Rect& target : result.wrongDetections)
			cv::rectangle(output, target, wrongDetectionColor, thickness);
		for (const Rect& target : result.ignoredDetections)
			cv::rectangle(output, target, ignoredDetectionColor, thickness);
		for (const Rect& target : result.missedDetections)
			cv::rectangle(output, target, missedDetectionColor, thickness);
		cv::imshow("Detections", output);
		int key = cv::waitKey(0);
		if (static_cast<char>(key) == 'q')
			return false;
	}
	return true;
}

TaskType getTaskType(const string& type) {
	if (type == "train")
		return TaskType::TRAIN;
	if (type == "test")
		return TaskType::TEST;
	if (type == "show")
		return TaskType::SHOW;
	throw invalid_argument("expected train/test/show, but was '" + (string)type + "'");
}

FeatureType getFeatureType(const string& type) {
	if (type == "fhog")
		return FeatureType::FHOG;
	if (type == "fastfhog")
		return FeatureType::FAST_FHOG;
	if (type == "gradhist")
		return FeatureType::GRADHIST;
	if (type == "fpdw")
		return FeatureType::FPDW;
	throw invalid_argument("expected fhog/gradhist/fpdw, but was '" + type + "'");
}

FeatureType getFeatureType(const ptree& config) {
	return getFeatureType(config.get<string>("type"));
}

FeatureParams getFeatureParams(const ptree& config) {
	FeatureParams parameters;
	parameters.windowSizeInCells.width = config.get<int>("windowWidthInCells");
	parameters.windowSizeInCells.height = config.get<int>("windowHeightInCells");
	parameters.cellSizeInPixels = config.get<int>("cellSizeInPixels");
	parameters.octaveLayerCount = config.get<int>("octaveLayerCount");
	parameters.widthScaleFactor = config.get<float>("widthScaleFactor");
	parameters.heightScaleFactor = config.get<float>("heightScaleFactor");
	return parameters;
}

TrainingParams getTrainingParams(const ptree& config) {
	TrainingParams parameters;
	parameters.mirrorTrainingData = config.get<bool>("mirrorTrainingData");
	parameters.maxNegatives = config.get<int>("maxNegatives");
	parameters.randomNegativesPerImage = config.get<int>("randomNegativesPerImage");
	parameters.maxHardNegativesPerImage = config.get<int>("maxHardNegativesPerImage");
	parameters.bootstrappingRounds = config.get<int>("bootstrappingRounds");
	parameters.negativeScoreThreshold = config.get<float>("negativeScoreThreshold");
	parameters.overlapThreshold = config.get<double>("overlapThreshold");
	parameters.C = config.get<double>("C");
	parameters.compensateImbalance = config.get<bool>("compensateImbalance");
	parameters.probabilistic = config.get<bool>("probabilistic");
	return parameters;
}

DetectionParams getDetectionParams(const ptree& config) {
	DetectionParams parameters;
	parameters.minWindowSizeInPixels.width = config.get<int>("minWindowWidthInPixels");
	parameters.minWindowSizeInPixels.height = config.get<int>("minWindowHeightInPixels");
	parameters.octaveLayerCount = config.get<int>("octaveLayerCount");
	parameters.approximatePyramid = config.get<bool>("approximatePyramid");
	parameters.nmsOverlapThreshold = config.get<double>("nmsOverlapThreshold");
	return parameters;
}

void printUsageInformation(string applicationName) {
	cout << "call: " << applicationName << " action directory images setcount configs" << endl;
	cout << "action: what the program should do" << endl;
	cout << "  train: train detector(s)" << endl;
	cout << "  test: test detector(s)" << endl;
	cout << "  show: show detection results of detector(s)" << endl;
	cout << "directory: directory to create or use for loading and storing SVM and evaluation data" << endl;
	cout << "images: DLib XML file of annotated images" << endl;
	cout << "setcount: number of subsets for cross-validation (1 to use all images at once)" << endl;
	cout << "configs: configuration file(s) for features, training and detection, depends on action" << endl;
	cout << "  train: [featureconfig trainingconfig] (only necessary if detector directory does not exist)" << endl;
	cout << "    featureconfig: configuration file containing feature parameters" << endl;
	cout << "    trainingconfig: configuration file containing training parameters" << endl;
	cout << "  test: detectionconfig" << endl;
	cout << "  show: detectionconfig [threshold]" << endl;
	cout << "    detectionconfig: configuration file containing detection parameters" << endl;
	cout << "    threshold: SVM score threshold (optional, defaults to 0.0)" << endl;
	cout << endl;
	cout << "Examples:" << endl;
	cout << "Create four detectors for cross-validation:" << endl << "  " << applicationName << " train mydetector images.xml 4 featureconfig trainingconfig" << endl;
	cout << "Train single detector in existing directory, re-using configs: " << endl << "  " << applicationName << " train mydetector images.xml 1" << endl;
	cout << "Evaluate detectors using cross-validation: " << endl << "  " << applicationName << " test mydetector images.xml 4 detectorconfig" << endl;
	cout << "Show detections using cross-validation: " << endl << "  " << applicationName << " show mydetector images.xml 4 detectorconfig 0.5" << endl;
}

int main(int argc, char** argv) {
	if (argc < 5 || argc > 7) {
		printUsageInformation(argv[0]);
		return 0;
	}
	TaskType taskType = getTaskType(argv[1]);
	path directory = argv[2];
	auto imageSource = make_shared<DlibImageSource>(argv[3]);
	int setCount = std::stoi(argv[4]);
	ptree featureConfig, trainingConfig, detectionConfig;

	if (taskType == TaskType::TRAIN) {
		if (argc == 5) { // re-use directory, read training and feature config contained in directory
			if (exists(directory)) {
				cout << "using existing directory '" << directory.string() << "'" << endl;
				read_info((directory / "featureparams").string(), featureConfig);
				read_info((directory / "trainingparams").string(), trainingConfig);
			} else {
				cerr << "directory '" << directory.string() << "' does not exist, exiting program" << endl;
				cerr << "(to create new directory, do specify training and feature configurations)" << endl;
				return 0;
			}
		} else if (argc == 7) { // create directory, copy training and feature config into directory
			if (exists(directory)) {
				cerr << "directory '" << directory.string() << "' already exists, exiting program" << endl;
				cerr << "(to re-use directory and its configuration, do not specify training and feature configurations)" << endl;
				return 0;
			} else {
				cout << "creating new directory '" << directory.string() << "'" << endl;
				create_directory(directory);
				read_info(argv[5], featureConfig);
				read_info(argv[6], trainingConfig);
				write_info((directory / "featureparams").string(), featureConfig);
				write_info((directory / "trainingparams").string(), trainingConfig);
			}
		} else {
			printUsageInformation(argv[0]);
			return 0;
		}
	} else {
		if ((taskType == TaskType::TEST && argc != 6)
				|| (taskType == TaskType::SHOW && argc < 6)) {
			printUsageInformation(argv[0]);
			return 0;
		}
		if (!exists(directory)) {
			cerr << "directory '" << directory.string() << "' does not exist, exiting program" << endl;
			return 0;
		}
		read_info((directory / "featureparams").string(), featureConfig);
		read_info((directory / "trainingparams").string(), trainingConfig);
		read_info(argv[5], detectionConfig);
	}
	FeatureType featureType = getFeatureType(featureConfig);
	FeatureParams featureParams = getFeatureParams(featureConfig);
	vector<LabeledImage> imageSet = getLabeledImages(imageSource, featureParams);

	if (taskType == TaskType::TRAIN) {
		TrainingParams trainingParams = getTrainingParams(trainingConfig);
		DetectorTrainer detectorTrainer(true, "  ");
		detectorTrainer.setTrainingParameters(trainingParams);
		setFeatures(detectorTrainer, featureType, featureParams);
		steady_clock::time_point start = steady_clock::now();
		cout << "train detector '" << directory.string() << "' on ";
		if (setCount == 1) { // train on all images
			cout << "all images" << endl;
			path svmFile = directory / "svm";
			if (exists(svmFile)) {
				cout << "  SVM file '" << svmFile.string() << "' already exists, skipping training" << endl;
			} else {
				detectorTrainer.train(imageSet);
				detectorTrainer.storeClassifier(svmFile.string());
			}
		} else { // train on subsets for cross-validation
			cout << setCount << " sets" << endl;
			vector<vector<LabeledImage>> subsets = getSubsets(imageSet, setCount);
			for (int testSetIndex = 0; testSetIndex < subsets.size(); ++testSetIndex) {
				cout << "training on subset " << (testSetIndex + 1) << endl;
				path svmFile = directory / ("svm" + std::to_string(testSetIndex + 1));
				if (exists(svmFile)) {
					cout << "  SVM file '" << svmFile.string() << "' already exists, skipping training" << endl;
				} else {
					detectorTrainer.train(getTrainingSet(subsets, testSetIndex));
					detectorTrainer.storeClassifier(svmFile.string());
				}
			}
		}
		steady_clock::time_point end = steady_clock::now();
		seconds trainingTime = duration_cast<seconds>(end - start);
		cout << "training finished after ";
		int seconds = trainingTime.count();
		if (seconds >= 60) {
			int minutes = seconds / 60;
			seconds = seconds % 60;
			if (minutes >= 60) {
				int hours = minutes / 60;
				minutes = minutes % 60;
				if (hours >= 24) {
					int days = hours / 24;
					hours = hours % 24;
					cout << days << " d ";
				}
				cout << hours << " h ";
			}
			cout << minutes << " min ";
		}
		cout << seconds << " sec " << endl;
	}
	else if (taskType == TaskType::TEST) {
		DetectionParams detectionParams = getDetectionParams(detectionConfig);
		string paramName = path(argv[5]).filename().replace_extension().string();
		DetectorTester tester(detectionParams.minWindowSizeInPixels);
		cout << "test detector '" << directory.string() << "' with parameters '" << paramName << "' on ";
		if (setCount == 1) // no cross-validation, test on all images at once
			cout << "all images" << endl;
		else // cross-validation, test on subsets
			cout << setCount << " sets" << endl;
		path evaluationDataFile = directory / ("evaluation_" + paramName);
		if (exists(evaluationDataFile)) {
			cout << "loading evaluation data from " << evaluationDataFile << endl;
			tester.loadData(evaluationDataFile.string());
		} else {
			if (setCount == 1) { // no cross-validation, test on all images at once
				path svmFile = directory / "svm";
				shared_ptr<AggregatedFeaturesDetector> detector = loadDetector(
						svmFile.string(), featureType, featureParams, detectionParams, -1.0f);
				tester.evaluate(*detector, imageSet);
			} else { // cross-validation, test on subsets
				vector<vector<LabeledImage>> subsets = getSubsets(imageSet, setCount);
				for (int testSetIndex = 0; testSetIndex < subsets.size(); ++testSetIndex) {
					cout << "testing on subset " << (testSetIndex + 1) << endl;
					path svmFile = directory / ("svm" + std::to_string(testSetIndex + 1));
					shared_ptr<AggregatedFeaturesDetector> detector = loadDetector(
							svmFile.string(), featureType, featureParams, detectionParams, -1.0f);
					tester.evaluate(*detector, subsets[testSetIndex]);
				}
			}
			tester.storeData(evaluationDataFile.string());
			path detCurveFile = directory / ("det_" + paramName);
			tester.writeDetCurve(detCurveFile.string());
		}
		DetectorEvaluationSummary summary = tester.getSummary();
		cout << "=== Evaluation summary ===" << endl;
		summary.writeTo(cout);
		path summaryFile = directory / ("summary_" + paramName);
		std::ofstream summaryFileStream(summaryFile.string());
		summary.writeTo(summaryFileStream);
		summaryFileStream.close();
	}
	else if (taskType == TaskType::SHOW) {
		float threshold = argc > 6 ? std::stof(argv[6]) : 0;
		DetectionParams detectionParams = getDetectionParams(detectionConfig);
		DetectorTester tester(detectionParams.minWindowSizeInPixels);
		if (setCount == 1) { // no cross-validation, show same detector on all images
			path svmFile = directory / "svm";
			shared_ptr<AggregatedFeaturesDetector> detector = loadDetector(
					svmFile.string(), featureType, featureParams, detectionParams, threshold);
			showDetections(tester, *detector, imageSet);
		} else { // cross-validation, show subset-detectors
			vector<vector<LabeledImage>> subsets = getSubsets(imageSet, setCount);
			for (int testSetIndex = 0; testSetIndex < subsets.size(); ++testSetIndex) {
				path svmFile = directory / ("svm" + std::to_string(testSetIndex + 1));
				shared_ptr<AggregatedFeaturesDetector> detector = loadDetector(
						svmFile.string(), featureType, featureParams, detectionParams, threshold);
				if (!showDetections(tester, *detector, subsets[testSetIndex]))
					break;
			}
		}
	}

	return 0;
}
