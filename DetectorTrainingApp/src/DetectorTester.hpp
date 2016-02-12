/*
 * DetectorTester.hpp
 *
 *  Created on: 23.10.2015
 *      Author: poschmann
 */

#ifndef DETECTORTESTER_HPP_
#define DETECTORTESTER_HPP_

#include "Annotations.hpp"
#include "LabeledImage.hpp"
#include "detection/SimpleDetector.hpp"
#include "imageio/RectLandmark.hpp"
#include "opencv2/core/core.hpp"
#include <chrono>
#include <memory>
#include <vector>

/**
 * Result of a detector evaluation.
 */
class DetectorEvaluationResult {
public:

	/**
	 * Creates a detector test result for a single image.
	 */
	static DetectorEvaluationResult single() {
		DetectorEvaluationResult result;
		result.images = 1;
		return result;
	}

	int getTruePositives() const {
		return truePositives;
	}

	int getFalsePositives() const {
		return falsePositives;
	}

	int getFalseNegatives() const {
		return falseNegatives;
	}

	double getPrecision() const {
		if (selected == 0)
			return 0;
		return static_cast<double>(truePositives) / static_cast<double>(selected);
	}

	double getRecall() const {
		if (relevant == 0)
			return 0;
		return static_cast<double>(truePositives) / static_cast<double>(relevant);
	}

	double getF1Measure() const {
		return getFMeasure(1);
	}

	double getFMeasure(double beta) const {
		double betaSquared = beta * beta;
		double precision = getPrecision();
		double recall = getRecall();
		return ((1 + betaSquared) * precision * recall) / (betaSquared * precision + recall);
	}

	std::chrono::milliseconds getAverageDetectionDuration() const {
		if (images == 0)
			return std::chrono::milliseconds::zero();
		return detectionDurationSum / images;
	}

	DetectorEvaluationResult& operator+=(const DetectorEvaluationResult& other) {
		images += other.images;
		truePositives += other.truePositives;
		falsePositives += other.falsePositives;
		falseNegatives += other.falseNegatives;
		selected += other.selected;
		relevant += other.relevant;
		detectionDurationSum += other.detectionDurationSum;
		return *this;
	}

protected:
	friend class DetectorTester;

	int images = 0;
	int truePositives = 0;
	int falsePositives = 0;
	int falseNegatives = 0;
	int selected = 0; // selected = true positives + false positives
	int relevant = 0; // relevant = true positives + false negatives
	std::chrono::milliseconds detectionDurationSum = std::chrono::milliseconds::zero();
};

/**
 * Result of a detection in an image after comparison with the ground truth.
 */
struct DetectionResult {
	std::vector<cv::Rect> correctDetections; ///< True positive detections.
	std::vector<cv::Rect> wrongDetections; ///< False positive detections.
	std::vector<cv::Rect> missedDetections; ///< False negative detections.
	std::vector<cv::Rect> ignoredDetections; ///< Ignored detections.
};

/**
 * Tester for detectors.
 */
class DetectorTester {
public:

	/**
	 * Constructs a new detector tester.
	 *
	 * @param[in] overlapThreshold Minimum overlap necessary to assign a detection to a ground truth bounding box.
	 */
	explicit DetectorTester(double overlapThreshold = 0.5);

	/**
	 * Detects targets inside an image using a detector.
	 *
	 * @param[in] detector Detector.
	 * @param[in] image Image.
	 * @param[in] landmarks Labeled bounding boxes that are either positive or should be ignored (neither positive, nor negative).
	 * @return Result containing correct, wrong, ignored and missed detections.
	 */
	DetectionResult detect(detection::SimpleDetector& detector,
			const cv::Mat& image, const std::vector<imageio::RectLandmark>& landmarks) const;

	/**
	 * Evaluates a detector on several images.
	 *
	 * @param[in] detector Detector that should be evaluated.
	 * @param[in] images Images with labeled bounding boxes.
	 * @return Evaluation result for the images.
	 */
	DetectorEvaluationResult evaluate(detection::SimpleDetector& detector, std::vector<LabeledImage>& images) const;

	/**
	 * Evaluates a detector on a single image.
	 *
	 * @param[in] detector Detector that should be evaluated.
	 * @param[in] image Image to detect targets in.
	 * @param[in] landmarks Labeled bounding boxes that are either positive or should be ignored (neither positive, nor negative).
	 * @return Evaluation result for the single image.
	 */
	DetectorEvaluationResult evaluate(detection::SimpleDetector& detector,
			const cv::Mat& image, const std::vector<imageio::RectLandmark>& landmarks) const;

private:

	enum class DetectionStatus : uint8_t { FALSE_POSITIVE, TRUE_POSITIVE, IGNORED };

	enum class PositiveStatus : uint8_t { FALSE_NEGATIVE, TRUE_POSITIVE };

	struct Status {
		std::vector<DetectionStatus> detectionStatus;
		std::vector<PositiveStatus> positiveStatus;
	};

	std::vector<cv::Rect> detectAndMeasureDuration(detection::SimpleDetector& detector,
			const cv::Mat& image, DetectorEvaluationResult& result) const;

	void measurePerformance(const std::vector<cv::Rect>& detections, const Annotations& annotations, DetectorEvaluationResult& result) const;

	Status compareWithGroundTruth(const std::vector<cv::Rect>& detections, const Annotations& annotations) const;

	cv::Mat createOverlapMatrix(const std::vector<cv::Rect>& detections, const std::vector<cv::Rect>& annotations) const;

	double computeOverlap(cv::Rect a, cv::Rect b) const;

	void computeStatus(cv::Mat& positiveOverlaps, cv::Mat& ignoreOverlaps, DetectorTester::Status& status) const;

	double overlapThreshold; ///< Minimum overlap necessary to assign a detection to a ground truth bounding box.
};

#endif /* DETECTORTESTER_HPP_ */
