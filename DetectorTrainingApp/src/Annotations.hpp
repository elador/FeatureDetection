/*
 * Annotations.hpp
 *
 *  Created on: 06.11.2015
 *      Author: poschmann
 */

#ifndef ANNOTATIONS_HPP_
#define ANNOTATIONS_HPP_

#include "imageio/LabeledImageSource.hpp"
#include "imageio/Landmark.hpp"
#include "imageio/RectLandmark.hpp"
#include "opencv2/core/core.hpp"
#include <vector>

/**
 * Annotated bounding boxes.
 *
 * The bounding boxes are either positive or fuzzy, but never negative. Fuzzy annotations are neither
 * positive, nor negative and should usually be ignored.
 */
class Annotations {
public:

	Annotations() = default;

	/**
	 * Constructs new annotations from the given landmarks.
	 *
	 * @param[in] landmarks Landmarks to construct annotations from.
	 * @param[in] minSize Minimum allowed size of positive bounding boxes (smaller ones are relabeled as fuzzy).
	 */
	explicit Annotations(const std::vector<std::shared_ptr<imageio::Landmark>>& landmarks, cv::Size minSize = cv::Size()) {
		extractBounds(landmarks, minSize);
	}

	/**
	 * Constructs new annotations from the given rectangular landmarks.
	 *
	 * @param[in] landmarks Landmarks to construct annotations from.
	 * @param[in] minSize Minimum allowed size of positive bounding boxes (smaller ones are relabeled as fuzzy).
	 */
	explicit Annotations(const std::vector<imageio::RectLandmark>& landmarks, cv::Size minSize = cv::Size()) {
		extractBounds(landmarks, minSize);
	}

private:

	void extractBounds(const std::vector<std::shared_ptr<imageio::Landmark>>& landmarks, cv::Size minSize) {
		for (const auto& landmark : landmarks) {
			cv::Rect bounds = getBounds(*landmark);
			nonNegatives.push_back(bounds);
			if (isFuzzy(*landmark, bounds, minSize))
				fuzzies.push_back(bounds);
			else
				positives.push_back(bounds);
		}
	}

	void extractBounds(const std::vector<imageio::RectLandmark>& landmarks, cv::Size minSize) {
		for (const auto& landmark : landmarks) {
			cv::Rect bounds = getBounds(landmark);
			nonNegatives.push_back(bounds);
			if (isFuzzy(landmark, bounds, minSize))
				fuzzies.push_back(bounds);
			else
				positives.push_back(bounds);
		}
	}

	cv::Rect getBounds(const imageio::Landmark& landmark) const {
		cv::Rect_<float> rect = landmark.getRect();
		int x = static_cast<int>(std::round(rect.tl().x));
		int y = static_cast<int>(std::round(rect.tl().y));
		int width = static_cast<int>(std::round(rect.br().x)) - x;
		int height = static_cast<int>(std::round(rect.br().y)) - y;
		return cv::Rect(x, y, width, height);
	}

	bool isFuzzy(const imageio::Landmark& landmark, cv::Rect bounds, cv::Size minSize) const {
		return nameStartsWithIgnore(landmark) || boundsAreTooSmall(bounds, minSize);
	}

	bool nameStartsWithIgnore(const imageio::Landmark& landmark) const {
		return landmark.getName().compare(0, 6, "ignore") == 0;
	}

	bool boundsAreTooSmall(cv::Rect bounds, cv::Size minSize) const {
		return bounds.width < minSize.width && bounds.height < minSize.height;
	}

public:

	std::vector<cv::Rect> nonNegatives; ///< Bounding boxes that are considered non-negative (positives and fuzzies).
	std::vector<cv::Rect> positives; ///< Bounding boxes that are considered positive.
	std::vector<cv::Rect> fuzzies; ///< Bounding boxes that are neither positive, nor negative.
};

#endif /* ANNOTATIONS_HPP_ */
