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

	cv::Mat image;
	std::vector<imageio::RectLandmark> landmarks;
};

#endif /* LABELEDIMAGE_HPP_ */
