/*
 * LabeledImage.hpp
 *
 *  Created on: 12.02.2016
 *      Author: poschmann
 */

#ifndef LABELEDIMAGE_HPP_
#define LABELEDIMAGE_HPP_

#include "imageio/Landmark.hpp"
#include "imageio/RectLandmark.hpp"
#include "opencv2/core/core.hpp"
#include <memory>
#include <vector>

/**
 * Image with landmarks.
 */
class LabeledImage {
public:

	LabeledImage(const cv::Mat& image, std::vector<std::shared_ptr<imageio::Landmark>> landmarks) : image(image), landmarks() {
		this->landmarks.reserve(landmarks.size());
		for (const std::shared_ptr<imageio::Landmark>& landmark : landmarks) {
			this->landmarks.emplace_back(landmark->getName(), landmark->getPosition2D(), landmark->getSize());
		}
	}

	/**
	 * Adjusts the sizes of the landmarks according to the given aspect ratio.
	 *
	 * If the aspect ratio of a landmark diverges from the given one, then either the width or height
	 * will be increased. The adjusted bounds will always contain the original bounds.
	 *
	 * @param[in] aspectRatio Aspect ratio of the window (width / height).
	 */
	void adjustSizes(double aspectRatio) {
		double aspectRatioInv = 1.0 / aspectRatio;
		std::vector<imageio::RectLandmark> adjustedLandmarks;
		for (imageio::RectLandmark& landmark : landmarks) {
			float width = landmark.getWidth();
			float height = landmark.getHeight();
			if (width < aspectRatio * height)
				width = aspectRatio * height;
			else if (width > aspectRatio * height)
				height = width * aspectRatioInv;
			adjustedLandmarks.emplace_back(landmark.getName(), landmark.getX(), landmark.getY(), width, height);
		}
		std::swap(landmarks, adjustedLandmarks);
	}

	/**
	 * Adjusts the width of the landmarks according to the given aspect ratio. The height remains unchanged.
	 *
	 * @param[in] aspectRatio Aspect ratio of the window (width / height).
	 */
	void adjustWidths(double aspectRatio) {
		std::vector<imageio::RectLandmark> adjustedLandmarks;
		for (imageio::RectLandmark& landmark : landmarks)
			adjustedLandmarks.emplace_back(landmark.getName(), landmark.getX(), landmark.getY(),
					aspectRatio * landmark.getHeight(), landmark.getHeight());
		std::swap(landmarks, adjustedLandmarks);
	}

	/**
	 * Adjusts the height of the landmarks according to the given aspect ratio. The width remains unchanged.
	 *
	 * @param[in] aspectRatio Aspect ratio of the window (width / height).
	 */
	void adjustHeights(double aspectRatio) {
		double aspectRatioInv = 1.0 / aspectRatio;
		std::vector<imageio::RectLandmark> adjustedLandmarks;
		for (imageio::RectLandmark& landmark : landmarks)
			adjustedLandmarks.emplace_back(landmark.getName(), landmark.getX(), landmark.getY(),
						landmark.getWidth(), aspectRatioInv * landmark.getWidth());
		std::swap(landmarks, adjustedLandmarks);
	}

	cv::Mat image;
	std::vector<imageio::RectLandmark> landmarks;
};

#endif /* LABELEDIMAGE_HPP_ */
