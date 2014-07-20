/*
 * OpenCVSiftFilter.hpp
 *
 *  Created on: 03.07.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef OPENCVSIFTFILTER_HPP_
#define OPENCVSIFTFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

#include "opencv2/nonfree/nonfree.hpp"

namespace superviseddescent { // We'll move it to imageprocessing once finished

/**
 * Filter that uses cv::SIFT to extract SIFT features of the given image
 * patch. No keypoint detection is performed. The center of the patch is
 * used as the center of where to extract the features.
 */
class OpenCVSiftFilter : public imageprocessing::ImageFilter {
public:

	/**
	 * Constructs a new SIFT filter that uses cv::SIFT to extract SIFT features.
	 */
	OpenCVSiftFilter();

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:
	cv::SIFT sift; ///< OpenCV object that manages the feature extraction
};

} /* namespace superviseddescent */
#endif /* OPENCVSIFTFILTER_HPP_ */
