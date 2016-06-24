/*
 * DetectorTrainingApp.cpp
 *
 *  Created on: 23.10.2015
 *      Author: poschmann
 */

#include "stacktrace.hpp"
#include "boost/filesystem.hpp"
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
using cv::Mat;
using cv::Rect;
using cv::Size;
using classification::SvmClassifier;
using detection::AggregatedFeaturesDetector;
using detection::NonMaximumSuppression;
using imageio::DlibImageSource;
using imageio::LabeledImageSource;
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

vector<LabeledImage> getLabeledImages(shared_ptr<LabeledImageSource> source) {
	vector<LabeledImage> images;
	while (source->next())
		images.emplace_back(source->getImage(), source->getLandmarks().getLandmarks());
	return images;
}

shared_ptr<AggregatedFeaturesDetector> createDetector(const shared_ptr<SvmClassifier>& svm,
		const shared_ptr<ImageFilter>& imageFilter, const shared_ptr<ImageFilter>& layerFilter,
		FeatureParams featureParams, shared_ptr<NonMaximumSuppression> nms, int octaveLayerCount) {
	if (imageFilter)
		return make_shared<AggregatedFeaturesDetector>(imageFilter, layerFilter,
				featureParams.cellSizeInPixels, featureParams.windowSizeInCells, octaveLayerCount,
				svm, nms, featureParams.widthScaleFactorInv(), featureParams.heightScaleFactorInv());
	else
		return make_shared<AggregatedFeaturesDetector>(layerFilter,
				featureParams.cellSizeInPixels, featureParams.windowSizeInCells, octaveLayerCount,
				svm, nms, featureParams.widthScaleFactorInv(), featureParams.heightScaleFactorInv());
}

shared_ptr<AggregatedFeaturesDetector> createApproximateDetector(const shared_ptr<SvmClassifier>& svm,
		const shared_ptr<ImageFilter>& imageFilter, const shared_ptr<ImageFilter>& layerFilter,
		FeatureParams featureParams, shared_ptr<NonMaximumSuppression> nms, int octaveLayerCount, vector<double> lambdas) {
	auto featurePyramid = ImagePyramid::createApproximated(octaveLayerCount, 0.5, 1.0, lambdas);
	if (imageFilter)
		featurePyramid->addImageFilter(imageFilter);
	featurePyramid->addLayerFilter(layerFilter);
	auto extractor = make_shared<AggregatedFeaturesExtractor>(featurePyramid,
			featureParams.windowSizeInCells, featureParams.cellSizeInPixels, true);
	return make_shared<AggregatedFeaturesDetector>(extractor, svm, nms,
			featureParams.widthScaleFactorInv(), featureParams.heightScaleFactorInv());
}

shared_ptr<ImageFilter> createFhogFilter(FeatureParams featureParams, bool fast) {
	if (fast) {
		return make_shared<FhogFilter>(featureParams.cellSizeInPixels, 9, false, true, 0.2f);
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

void printTestSummary(DetectorEvaluationSummary summary) {
	cout << "Average time: " << summary.avgTime.count() << " ms" << endl;
	cout << "Default FPPI rate: " << summary.defaultFppiRate << endl;
	cout << "Default miss rate: " << summary.defaultMissRate << endl;
	cout << "Miss rate at 1 FPPI: " << summary.missRateAtFppi0 << " (threshold " << summary.thresholdAtFppi0 << ")" << endl;
	cout << "Miss rate at 0.1 FPPI: " << summary.missRateAtFppi1 << " (threshold " << summary.thresholdAtFppi1 << ")" << endl;
	cout << "Miss rate at 0.01 FPPI: " << summary.missRateAtFppi2 << " (threshold " << summary.thresholdAtFppi2 << ")" << endl;
	cout << "Log-average miss rate: " << summary.avgMissRate << endl;
}

void printTestSummary(const string& title, DetectorEvaluationSummary summary) {
	cout << "=== " << title << " ===" << endl;
	printTestSummary(summary);
}


enum class TaskType { TRAIN, TRAIN_FULL, TEST, TEST_APPROXIMATE, SHOW, SHOW_APPROXIMATE };
enum class FeatureType { FHOG, FAST_FHOG, GRADHIST, FPDW };

void storeParameters(const string& filename, TrainingParams trainingParams, FeatureParams featureParams, FeatureType featureType,
		int octaveLayerCountForDetection, double nmsOverlapThreshold) {
	std::ofstream file(filename);
	file << "TrainingParams:\n";
	file << "mirrorTrainingData" << trainingParams.mirrorTrainingData << '\n';
	file << "maxNegatives" << trainingParams.maxNegatives << '\n';
	file << "randomNegativesPerImage = " << trainingParams.randomNegativesPerImage << '\n';
	file << "maxHardNegativesPerImage = " << trainingParams.maxHardNegativesPerImage << '\n';
	file << "bootstrappingRounds = " << trainingParams.bootstrappingRounds << '\n';
	file << "negativeScoreThreshold = " << trainingParams.negativeScoreThreshold << '\n';
	file << "overlapThreshold = " << trainingParams.overlapThreshold << '\n';
	file << "C = " << trainingParams.C << '\n';
	file << "compensateImbalance = " << (trainingParams.compensateImbalance ? "true" : "false") << '\n';
	file << '\n';
	file << "FeatureParams:\n";
	file << "type = ";
	if (featureType == FeatureType::FHOG)
		file << "fhog";
	else if (featureType == FeatureType::FAST_FHOG)
		file << "fastfhog";
	else if (featureType == FeatureType::GRADHIST)
		file << "gradhist";
	else if (featureType == FeatureType::FPDW)
		file << "fpdw";
	else
		file << "unknown";
	file << '\n';
	file << "windowWidthInCells = " << featureParams.windowSizeInCells.width << '\n';
	file << "windowHeightInCells = " << featureParams.windowSizeInCells.height << '\n';
	file << "cellSizeInPixels = " << featureParams.cellSizeInPixels << '\n';
	file << "octaveLayerCount = " << featureParams.octaveLayerCount << '\n';
	file << "widthScaleFactor = " << featureParams.widthScaleFactor << '\n';
	file << "heightScaleFactor = " << featureParams.heightScaleFactor << '\n';
	file << '\n';
	file << "DetectionParams:\n";
	file << "octaveLayerCount = " << octaveLayerCountForDetection << '\n';
	file << "nmsOverlapThreshold = " << nmsOverlapThreshold << '\n';
	file.close();
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

shared_ptr<AggregatedFeaturesDetector> loadDetector(const string& filename, FeatureType featureType, FeatureParams featureParams,
		shared_ptr<NonMaximumSuppression> nms, int octaveLayerCount, bool approximate, float threshold = 0) {
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
	if (approximate)
		return createApproximateDetector(svm, imageFilter, layerFilter, featureParams, nms, octaveLayerCount, lambdas);
	return createDetector(svm, imageFilter, layerFilter, featureParams, nms, octaveLayerCount);
}

TaskType getTaskType(const string& type) {
	if (type == "train")
		return TaskType::TRAIN;
	if (type == "train-full")
		return TaskType::TRAIN_FULL;
	if (type == "test")
		return TaskType::TEST;
	if (type == "test-approximate")
		return TaskType::TEST_APPROXIMATE;
	if (type == "show")
		return TaskType::SHOW;
	if (type == "show-approximate")
		return TaskType::SHOW_APPROXIMATE;
	throw invalid_argument("expected train/test/test-approximate/show/show-approximate, but was '" + (string)type + "'");
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

int main(int argc, char** argv) {
	if (argc != 6) {
		cout << "call: ./DetectorTrainingApp action images setcount features directory" << endl;
		cout << "action: what the program should do" << endl;
		cout << "  train: train classifiers on subsets for cross-validation" << endl;
		cout << "  train-full: train classifiers on subsets for cross-validation and one classifier on all images" << endl;
		cout << "  test: test classifiers on subsets using cross-validation" << endl;
		cout << "  test-approximate: test approximated classifiers on subsets using cross-validation" << endl;
		cout << "  show: show detection results of classifiers on subsets" << endl;
		cout << "  show-approximate: show detection results of approximated classifiers on subsets" << endl;
		cout << "images: DLib XML file of annotated images" << endl;
		cout << "setcount: number of subsets for cross-validation (1 to use all images)" << endl;
		cout << "features: type of features to use" << endl;
		cout << "  fhog: FHOG descriptor" << endl;
		cout << "  fastfhog: fast FHOG descriptor (almost no difference in descriptor to fhog)" << endl;
		cout << "  gradhist: gradient histogram" << endl;
		cout << "  fpdw: features proposed in the paper of the \"Fastest Pedestrian Detector in the West\"" << endl;
		cout << "directory: directory to create or use for loading and storing SVM and evaluation data" << endl;
		return 0;
	}
	TaskType taskType = getTaskType(argv[1]);
	string imagesFilename = argv[2];
	int setCount = std::stoi(argv[3]);
	FeatureType featureType = getFeatureType(argv[4]);
	path directory = argv[5];

	TrainingParams trainingParams;
	trainingParams.mirrorTrainingData = true;
	trainingParams.maxNegatives = 50000;
	trainingParams.randomNegativesPerImage = 20;
	trainingParams.maxHardNegativesPerImage = 100;
	trainingParams.bootstrappingRounds = 3;
	trainingParams.negativeScoreThreshold = -1.0f;
	trainingParams.overlapThreshold = 0.3;
	trainingParams.C = 1;
	trainingParams.compensateImbalance = true;

	FeatureParams featureParams;
	featureParams.windowSizeInCells = Size(5, 7);
	featureParams.cellSizeInPixels = 7;
	featureParams.octaveLayerCount = 10;
	featureParams.widthScaleFactor = 1.0f;
	featureParams.heightScaleFactor = 1.0f;

	int octaveLayerCountForDetection = 5;
	shared_ptr<NonMaximumSuppression> nms = make_shared<NonMaximumSuppression>(0.3, NonMaximumSuppression::MaximumType::MAX_SCORE);
	vector<LabeledImage> imageSet = getLabeledImages(make_shared<DlibImageSource>(imagesFilename));

	if (taskType == TaskType::TRAIN) {
		if (exists(directory)) {
			cerr << "directory " << directory << " does already exist, exiting program" << endl;
			return 0;
		}
		create_directory(directory);
	} else if (!exists(directory)) {
		cerr << "directory " << directory << " does not exist, exiting program" << endl;
		return 0;
	}

	if (taskType == TaskType::TRAIN || taskType == TaskType::TRAIN_FULL) {
		DetectorTrainer detectorTrainer(true, "  ");
		detectorTrainer.setTrainingParameters(trainingParams);
		setFeatures(detectorTrainer, featureType, featureParams);
		steady_clock::time_point start = steady_clock::now();
		cout << "train detector on ";
		if (setCount == 1) { // no cross-validation, train on all images
			cout << "all images" << endl;
			detectorTrainer.train(imageSet);
			path svmFile = directory / "svm";
			detectorTrainer.storeClassifier(svmFile.string());
		} else { // cross-validation, train on subsets
			cout << setCount << " sets" << endl;
			vector<vector<LabeledImage>> subsets = getSubsets(imageSet, setCount);
			for (int testSetIndex = 0; testSetIndex < subsets.size(); ++testSetIndex) {
				cout << "train on subset " << (testSetIndex + 1) << endl;
				detectorTrainer.train(getTrainingSet(subsets, testSetIndex));
				path svmFile = directory / ("svm" + std::to_string(testSetIndex + 1));
				detectorTrainer.storeClassifier(svmFile.string());
			}
			if (taskType == TaskType::TRAIN_FULL) {
				cout << "train on all images" << endl;
				detectorTrainer.train(imageSet);
				path svmFile = directory / "svm";
				detectorTrainer.storeClassifier(svmFile.string());
			}
		}
		steady_clock::time_point end = steady_clock::now();
		seconds trainingTime = duration_cast<seconds>(end - start);
		cout << "training finished after ";
		if (trainingTime.count() >= 60)
			cout << (trainingTime.count() / 60) << " min ";
		cout << (trainingTime.count() % 60) << " sec" << endl;
		path parameterFile = directory / "parameters";
		storeParameters(parameterFile.string(),
				trainingParams, featureParams, featureType, octaveLayerCountForDetection, nms->getOverlapThreshold());
	}
	else if (taskType == TaskType::TEST || taskType == TaskType::TEST_APPROXIMATE) {
		DetectorTester tester;
		bool approximate = taskType == TaskType::TEST_APPROXIMATE;
		string suffix = approximate ? "_approximate" : "";
		cout << "test " << (approximate ? "approximated " : "") << "detector on ";
		if (setCount == 1) // no cross-validation, test on all images at once
			cout << "all images" << endl;
		else // cross-validation, test on subsets
			cout << setCount << " sets" << endl;
		path evaluationDataFile = directory / ("evaluation" + suffix);
		if (exists(evaluationDataFile)) {
			cout << "loading evaluation data from " << evaluationDataFile << endl;
			tester.loadData(evaluationDataFile.string());
		} else {
			if (setCount == 1) { // no cross-validation, test on all images at once
				path svmFile = directory / "svm";
				shared_ptr<AggregatedFeaturesDetector> detector = loadDetector(svmFile.string(),
						featureType, featureParams, nms, octaveLayerCountForDetection, approximate, -1.0f);
				tester.evaluate(*detector, imageSet);
			} else { // cross-validation, test on subsets
				vector<vector<LabeledImage>> subsets = getSubsets(imageSet, setCount);
				for (int testSetIndex = 0; testSetIndex < subsets.size(); ++testSetIndex) {
					cout << "test on subset " << (testSetIndex + 1) << endl;
					path svmFile = directory / ("svm" + std::to_string(testSetIndex + 1));
					shared_ptr<AggregatedFeaturesDetector> detector = loadDetector(svmFile.string(),
							featureType, featureParams, nms, octaveLayerCountForDetection, approximate, -1.0f);
					tester.evaluate(*detector, subsets[testSetIndex]);
				}
			}
			tester.storeData(evaluationDataFile.string());
			path detCurveFile = directory / ("det" + suffix);
			tester.writeDetCurve(detCurveFile.string());
		}
		printTestSummary("Evaluation summary", tester.getSummary());
	}
	else if (taskType == TaskType::SHOW || taskType == TaskType::SHOW_APPROXIMATE) {
		DetectorTester tester;
		bool approximate = taskType == TaskType::SHOW_APPROXIMATE;
		if (setCount == 1) { // no cross-validation, show same detector on all images
			path svmFile = directory / "svm";
			shared_ptr<AggregatedFeaturesDetector> detector = loadDetector(svmFile.string(),
					featureType, featureParams, nms, octaveLayerCountForDetection, approximate);
			showDetections(tester, *detector, imageSet);
		} else { // cross-validation, show subset-detectors
			vector<vector<LabeledImage>> subsets = getSubsets(imageSet, setCount);
			for (int testSetIndex = 0; testSetIndex < subsets.size(); ++testSetIndex) {
				path svmFile = directory / ("svm" + std::to_string(testSetIndex + 1));
				shared_ptr<AggregatedFeaturesDetector> detector = loadDetector(svmFile.string(),
						featureType, featureParams, nms, octaveLayerCountForDetection, approximate);
				if (!showDetections(tester, *detector, subsets[testSetIndex]))
					break;
			}
		}
	}

	return 0;
}
