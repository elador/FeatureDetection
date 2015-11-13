/*
 * DetectorTrainingApp.cpp
 *
 *  Created on: 23.10.2015
 *      Author: poschmann
 */

#include "stacktrace.hpp"
#include "DetectorTester.hpp"
#include "DetectorTrainer.hpp"
#include "DilatedLabeledImageSource.hpp"
#include "classification/SvmClassifier.hpp"
#include "imageio/DlibImageSource.hpp"
#include "imageprocessing/CellBasedPyramidFeatureExtractor.hpp"
#include "imageprocessing/CompleteExtendedHogFilter.hpp"
#include "imageprocessing/ExtendedHogFeatureExtractor.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <memory>

using cv::Mat;
using cv::Rect;
using cv::Size;
using classification::SvmClassifier;
using detection::AggregatedFeaturesDetector;
using detection::NonMaximumSuppression;
using imageio::DlibImageSource;
using imageio::LabeledImageSource;
using imageprocessing::CellBasedPyramidFeatureExtractor;
using imageprocessing::CompleteExtendedHogFilter;
using imageprocessing::ExtendedHogFeatureExtractor;
using imageprocessing::GrayscaleFilter;
using imageprocessing::ImagePyramid;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;

shared_ptr<LabeledImageSource> getDilatedAnnotations(shared_ptr<LabeledImageSource> source, FeatureParams featureParams) {
	float dilationInPixels = 2.0f;
	float dilationInCells = dilationInPixels / featureParams.cellSizeInPixels;
	float widthScale = (featureParams.windowSizeInCells.width + dilationInCells) / featureParams.windowSizeInCells.width;
	float heightScale = (featureParams.windowSizeInCells.height + dilationInCells) / featureParams.windowSizeInCells.height;
	return make_shared<DilatedLabeledImageSource>(source, widthScale, heightScale);
}

shared_ptr<CompleteExtendedHogFilter> createEhogFilter(FeatureParams featureParams) {
	return make_shared<CompleteExtendedHogFilter>(featureParams.cellSizeInPixels, 18, true, true, false, true, 0.2);
}

void setEhogFeatures(DetectorTrainer& trainer, FeatureParams featureParams) {
	shared_ptr<CompleteExtendedHogFilter> hogFilter = createEhogFilter(featureParams);
	trainer.setFeatures(featureParams, hogFilter, make_shared<GrayscaleFilter>());
}

shared_ptr<AggregatedFeaturesDetector> loadEhogDetector(const string& filename,
		FeatureParams featureParams, shared_ptr<NonMaximumSuppression> nms, int octaveLayerCount) {
	shared_ptr<CompleteExtendedHogFilter> hogFilter = createEhogFilter(featureParams);
	std::ifstream stream(filename);
	shared_ptr<SvmClassifier> svm = SvmClassifier::load(stream);
	stream.close();
	return make_shared<AggregatedFeaturesDetector>(make_shared<GrayscaleFilter>(), hogFilter,
			featureParams.cellSizeInPixels, featureParams.windowSizeInCells, octaveLayerCount, svm, nms);
}

void showDetections(const DetectorTester& tester, AggregatedFeaturesDetector& detector, LabeledImageSource& source) {
	Mat output;
	cv::Scalar correctDetectionColor(0, 255, 0);
	cv::Scalar wrongDetectionColor(0, 0, 255);
	cv::Scalar ignoredDetectionColor(255, 204, 0);
	cv::Scalar missedDetectionColor(0, 153, 255);
	int thickness = 2;
	source.reset();
	while (source.next()) {
		DetectionResult result = tester.detect(detector, source.getImage(), source.getLandmarks().getLandmarks());
		source.getImage().copyTo(output);
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
			break;
	}
}

Mat drawEhogFeatures(const Mat& hog, size_t bins, bool fullHog) { // TODO remove or move to hog filter...
	size_t cellRows = hog.rows;
	size_t cellCols = hog.cols;
	cv::Scalar white(1.0);
	int thickness = 1;
	vector<Mat> bars(bins);
	bars[0] = Mat::zeros(20, 20, CV_32FC1);
	cv::line(bars[0], cv::Point(10, 10 - 8), cv::Point(10, 10 + 8), white, thickness);
	cv::Point2f center(10, 10);
	for (size_t bin = 1; bin < bins; ++bin) {
		double angle = bin * 180 / bins; // positive angle is clockwise, because y-axis points downwards
		Mat rotation = cv::getRotationMatrix2D(center, -angle, 1.0); // expects positive angle to be counter-clockwise
		cv::warpAffine(bars[0], bars[bin], rotation, bars[0].size(), cv::INTER_CUBIC);
	}
	Mat tmp = Mat::zeros(20 * cellRows, 20 * cellCols, CV_32FC1);
	for (size_t cellRow = 0; cellRow < cellRows; ++cellRow) {
		for (size_t cellCol = 0; cellCol < cellCols; ++cellCol) {
			const float* histogramValues = hog.ptr<float>(cellRow, cellCol);
			for (size_t bin = 0; bin < bins; ++bin) {
				float weight;
				if (fullHog)
					weight = histogramValues[bin] + histogramValues[bin + bins] + histogramValues[bin + 2 * bins];
				else
					weight = histogramValues[bin];
				if (weight > 0) {
					Mat cell(tmp, Rect(20 * cellCol, 20 * cellRow, 20, 20));
					cell = cv::max(cell, weight * bars[bin]);
				}
			}
		}
	}
	double max;
	cv::minMaxIdx(tmp, nullptr, &max);
	float threshold = static_cast<float>(max);
	Mat image(tmp.rows, tmp.cols, CV_8UC3);
	for (int row = 0; row < tmp.rows; ++row) {
		const float* tmpValues = tmp.ptr<float>(row);
		cv::Vec3b* outputValues = image.ptr<cv::Vec3b>(row);
		for (int col = 0; col < tmp.cols; ++col) {
			uchar value = cv::saturate_cast<uchar>(std::round(255 * tmpValues[col] / threshold));
			outputValues[col][0] = value;
			outputValues[col][1] = value;
			outputValues[col][2] = value;
		}
	}
	return image;
}

void printTestResult(DetectorEvaluationResult result) {
	cout << "F-Measure: " << result.getF1Measure() << endl;
	cout << "Precision: " << result.getPrecision() << endl;
	cout << "Recall: " << result.getRecall() << endl;
	cout << "True positives: " << result.getTruePositives() << endl;
	cout << "False positives: " << result.getFalsePositives() << endl;
	cout << "False negatives: " << result.getFalseNegatives() << endl;
	cout << "Average time: " << result.getAverageDetectionDuration().count() << " ms" << endl;
}

void printTestResult(const string& title, DetectorEvaluationResult result) {
	cout << "=== " << title << " ===" << endl;
	printTestResult(result);
}

int main(int argc, char** argv) {
	bool testOnly = argc > 1 && string(argv[1]) == "testonly";

	TrainingParams trainingParams;
	trainingParams.negativeScoreThreshold = -0.5;
	trainingParams.overlapThreshold = 0.3;
	trainingParams.C = 10;
	trainingParams.compensateImbalance = true;
	FeatureParams featureParams{Size(5, 7), 8, 10}; // (width, height), cellsize, octave
	shared_ptr<NonMaximumSuppression> nms = make_shared<NonMaximumSuppression>(0.3, NonMaximumSuppression::MaximumType::WEIGHTED_AVERAGE);

	auto trainingImages = getDilatedAnnotations(
			make_shared<DlibImageSource>("/home/poschmann/Bilder/FEI_Face_Database/training2/training.xml"), featureParams);
	auto testingImages = getDilatedAnnotations(
			make_shared<DlibImageSource>("/home/poschmann/Bilder/heads/testing/testing.xml"), featureParams);

	shared_ptr<AggregatedFeaturesDetector> detector;
	if (testOnly) {
		detector = loadEhogDetector("svmdata", featureParams, nms, 5);
	} else {
		DetectorTrainer trainer;
		trainer.setTrainingParameters(trainingParams);
		setEhogFeatures(trainer, featureParams);
		trainer.train(*trainingImages);
		trainer.storeClassifier("svmdata");

		Mat w = drawEhogFeatures(trainer.getWeightVector(), 9, true);
		cv::imshow("FHOG", w);

		detector = trainer.getDetector(nms, 5);
	}

	DetectorTester tester;
	printTestResult("Evaluation on training set", tester.evaluate(*detector, *trainingImages));
	printTestResult("Evaluation on testing set", tester.evaluate(*detector, *testingImages));
	showDetections(tester, *detector, *testingImages);

	return 0;
}
