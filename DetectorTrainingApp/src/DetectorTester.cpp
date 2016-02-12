/*
 * DetectorTester.cpp
 *
 *  Created on: 23.10.2015
 *      Author: poschmann
 */

#include "Annotations.hpp"
#include "DetectorTester.hpp"
#include "imageprocessing/VersionedImage.hpp"

using cv::Mat;
using cv::Rect;
using detection::SimpleDetector;
using imageio::RectLandmark;
using imageprocessing::VersionedImage;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;
using std::make_shared;
using std::shared_ptr;
using std::vector;

DetectorTester::DetectorTester(double overlapThreshold) : overlapThreshold(overlapThreshold) {}

DetectionResult DetectorTester::detect(SimpleDetector& detector, const Mat& image, const vector<RectLandmark>& landmarks) const {
	vector<Rect> detections = detector.detect(image);
	Annotations annotations(landmarks);
	Status status = compareWithGroundTruth(detections, annotations);
	DetectionResult result;
	for (int i = 0; i < detections.size(); ++i) {
		DetectionStatus s = status.detectionStatus[i];
		if (s == DetectionStatus::TRUE_POSITIVE)
			result.correctDetections.push_back(detections[i]);
		else if (s == DetectionStatus::FALSE_POSITIVE)
			result.wrongDetections.push_back(detections[i]);
		else if (s == DetectionStatus::IGNORED)
			result.ignoredDetections.push_back(detections[i]);
	}
	for (int i = 0; i < annotations.positives.size(); ++i) {
		PositiveStatus s = status.positiveStatus[i];
		if (s == PositiveStatus::FALSE_NEGATIVE)
			result.missedDetections.push_back(annotations.positives[i]);
	}
	return result;
}

DetectorEvaluationResult DetectorTester::evaluate(SimpleDetector& detector, vector<LabeledImage>& images) const {
	DetectorEvaluationResult result;
	for (const LabeledImage& image : images)
		result += evaluate(detector, image.image, image.landmarks);
	return result;
}

DetectorEvaluationResult DetectorTester::evaluate(SimpleDetector& detector,
		const Mat& image, const vector<RectLandmark>& landmarks) const {
	DetectorEvaluationResult result = DetectorEvaluationResult::single();
	vector<Rect> detections = detectAndMeasureDuration(detector, image, result);
	measurePerformance(detections, Annotations(landmarks), result);
	return result;
}

vector<Rect> DetectorTester::detectAndMeasureDuration(SimpleDetector& detector,
		const Mat& image, DetectorEvaluationResult& result) const {
	steady_clock::time_point start = steady_clock::now();
	vector<Rect> detections = detector.detect(image);
	steady_clock::time_point end = steady_clock::now();
	result.detectionDurationSum = duration_cast<milliseconds>(end - start);
	return detections;
}

void DetectorTester::measurePerformance(const vector<Rect>& detections, const Annotations& annotations, DetectorEvaluationResult& result) const {
	result.relevant += annotations.positives.size();
	result.selected += detections.size();
	Status status = compareWithGroundTruth(detections, annotations);
	for (DetectionStatus s : status.detectionStatus) {
		if (s == DetectionStatus::TRUE_POSITIVE)
			++result.truePositives;
		else if (s == DetectionStatus::FALSE_POSITIVE)
			++result.falsePositives;
		else if (s == DetectionStatus::IGNORED)
			--result.selected;
	}
	for (PositiveStatus s : status.positiveStatus) {
		if (s == PositiveStatus::FALSE_NEGATIVE)
			++result.falseNegatives;
	}
}

DetectorTester::Status DetectorTester::compareWithGroundTruth(const vector<Rect>& detections, const Annotations& annotations) const {
	DetectorTester::Status status;
	status.detectionStatus.resize(detections.size());
	status.positiveStatus.resize(annotations.positives.size());
	Mat positiveOverlaps = createOverlapMatrix(detections, annotations.positives);
	Mat ignoreOverlaps = createOverlapMatrix(detections, annotations.fuzzies);
	computeStatus(positiveOverlaps, ignoreOverlaps, status);
	return status;
}

Mat DetectorTester::createOverlapMatrix(const vector<Rect>& detections, const vector<Rect>& annotations) const {
	Mat overlaps(detections.size(), annotations.size(), CV_64FC1);
	for (int row = 0; row < detections.size(); ++row) {
		for (int col = 0; col < annotations.size(); ++col)
			overlaps.at<double>(row, col) = computeOverlap(detections[row], annotations[col]);
	}
	return overlaps;
}

double DetectorTester::computeOverlap(Rect a, Rect b) const {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}

void DetectorTester::computeStatus(Mat& positiveOverlaps, Mat& ignoreOverlaps, DetectorTester::Status& status) const {
	double maxPositiveOverlap, maxIgnoreOverlap;
	cv::Point positiveIndices, ignoreIndices;
	cv::minMaxLoc(positiveOverlaps, nullptr, &maxPositiveOverlap, nullptr, &positiveIndices);
	cv::minMaxLoc(ignoreOverlaps, nullptr, &maxIgnoreOverlap, nullptr, &ignoreIndices);
	while (std::max(maxPositiveOverlap, maxIgnoreOverlap) >= overlapThreshold) {
		if (maxPositiveOverlap >= maxIgnoreOverlap) {
			status.detectionStatus[positiveIndices.y] = DetectionStatus::TRUE_POSITIVE;
			status.positiveStatus[positiveIndices.x] = PositiveStatus::TRUE_POSITIVE;
			positiveOverlaps(cv::Range::all(), cv::Range(positiveIndices.x, positiveIndices.x + 1)) = 0.0;
			positiveOverlaps(cv::Range(positiveIndices.y, positiveIndices.y + 1), cv::Range::all()) = 0.0;
			ignoreOverlaps(cv::Range(positiveIndices.y, positiveIndices.y + 1), cv::Range::all()) = 0.0;
			cv::minMaxLoc(positiveOverlaps, nullptr, &maxPositiveOverlap, nullptr, &positiveIndices);
		} else {
			status.detectionStatus[ignoreIndices.y] = DetectionStatus::IGNORED;
			ignoreOverlaps(cv::Range::all(), cv::Range(ignoreIndices.x, ignoreIndices.x + 1)) = 0.0;
			ignoreOverlaps(cv::Range(ignoreIndices.y, ignoreIndices.y + 1), cv::Range::all()) = 0.0;
			positiveOverlaps(cv::Range(ignoreIndices.y, ignoreIndices.y + 1), cv::Range::all()) = 0.0;
			cv::minMaxLoc(ignoreOverlaps, nullptr, &maxIgnoreOverlap, nullptr, &ignoreIndices);
		}
	}
}
