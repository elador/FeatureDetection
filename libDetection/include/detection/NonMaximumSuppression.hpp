/*
 * NonMaximumSuppression.hpp
 *
 *  Created on: 28.10.2015
 *      Author: poschmann
 */

#ifndef DETECTION_NONMAXIMUMSUPPRESSION_HPP_
#define DETECTION_NONMAXIMUMSUPPRESSION_HPP_

#include "opencv2/core/core.hpp"
#include <vector>

namespace detection {

/**
 * Detection of an object.
 */
struct Detection {
	float score;
	cv::Rect bounds;
};

/**
 * Non-maximum suppression for eliminating redundant detections of the same object.
 */
class NonMaximumSuppression {
public:

	/** Type of the maximum computation per cluster. */
	enum class MaximumType {
		MAX_SCORE, ///< Take the detection with the highest score per cluster.
		AVERAGE, ///< Compute the average of the cluster.
		WEIGHTED_AVERAGE ///< Compute the average of the cluster, weighted by the score.
	};

	/**
	 * Constructs a new non-maximum suppression.
	 *
	 * @param[in] overlapThreshold Maximum allowed overlap between two detections, typically between 0.3 and 0.5.
	 * @param[in] maximumType Type of the maximum computation per cluster.
	 */
	explicit NonMaximumSuppression(double overlapThreshold, MaximumType maximumType = MaximumType::MAX_SCORE);

	/**
	 * Eliminates redundant detections of the same object.
	 *
	 * @param[in] candidates Distinct bounding boxes around the detected objects with their score.
	 * @return Non-redundant detections.
	 */
	std::vector<Detection> eliminateRedundantDetections(std::vector<Detection> candidates) const;

	/**
	 * @return Maximum allowed overlap between two detections (everything closer is regarded as the same object).
	 */
	double getOverlapThreshold() const;

	/**
	 * @return Type of the maximum computation per cluster.
	 */
	MaximumType getMaximumType() const;

private:

	/**
	 * Sorts the detections by their score in ascending order.
	 *
	 * @param[in,out] candidates Detections to sort.
	 */
	void sortByScore(std::vector<Detection>& candidates) const;

	/**
	 * Clusters redundant detections according to the overlap to their best scoring detection.
	 *
	 * @param[in] candidates Redundant detections sorted by their score in ascending order. Will be empty afterwards.
	 * @return Clustered detections, sorted by score, where all but one detection of each cluster are redundant.
	 */
	std::vector<std::vector<Detection>> cluster(std::vector<Detection>& candidates) const;

	/**
	 * Extracts all bounding boxes that overlap with the given detection. The overlapping bounding boxes are
	 * moved into a new vector.
	 *
	 * @param[in] detection Bounding box around the detected object with its score.
	 * @param[in,out] candidates Bounding boxes that may overlap with the given detection.
	 * @return Bounding boxes that overlap with the given detection.
	 */
	std::vector<Detection> extractOverlappingDetections(Detection detection, std::vector<Detection>& candidates) const;

	/**
	 * Computes the overlap between two rectangles. Overlap is defined as intersection divided by union.
	 *
	 * @param[in] a First rectangle.
	 * @param[in] b Second rectangle.
	 * @return Overlap between the two rectangles.
	 */
	double computeOverlap(cv::Rect a, cv::Rect b) const;

	/**
	 * Determines the maxima of the given clusters of detections.
	 *
	 * @param[in] cluster Clustered detections, where the detections of each cluster are ordered by their score in descending order.
	 * @return Maxima of the clusters.
	 */
	std::vector<Detection> getMaxima(const std::vector<std::vector<Detection>>& clusters) const;

	/**
	 * Determines the maximum of a given cluster of detections.
	 *
	 * @param[in] cluster Detections of the cluster, ordered by their score in descending order.
	 * @return Maximum of the cluster.
	 */
	Detection getMaximum(const std::vector<Detection>& cluster) const;

	double overlapThreshold; ///< Maximum allowed overlap between two detections (everything closer is regarded as the same object).
	MaximumType maximumType; ///< Type of the maximum computation per cluster.
};

} /* namespace detection */

#endif /* DETECTION_NONMAXIMUMSUPPRESSION_HPP_ */
