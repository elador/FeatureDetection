/*
 * GradientFilter.hpp
 *
 *  Created on: 08.10.2015
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_GRADIENTFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_GRADIENTFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {
namespace filtering {

/**
 * Image filter that computes gradients for each pixel.
 *
 * The resulting image has the same depth as the input image, but two channels for each of the input channels, one
 * for the gradient of x and one for y. If it is of an unsigned data type, then the neutral value is 127 for CV_8U
 * and 32767 for CV_16U. The derivative filter kernel is normalized.
 */
class GradientFilter : public ImageFilter {
public:

	/**
	 * Constructs a new gradient filter.
	 *
	 * @param[in] kernelSize The size of the derivative kernel. Must be positive and odd or CV_SCHARR.
	 */
	explicit GradientFilter(int kernelSize);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	void computeKernelScale();

	/**
	 * Computes the x and y gradients for each image channel.
	 *
	 * @param[in] image The image to compute the gradients of.
	 * @return Two gradient images, the first contains the x gradients and the second the y gradients.
	 */
	std::vector<cv::Mat> computeGradients(const cv::Mat& image) const;

	/**
	 * Merges the gradient images into one image.
	 *
	 * @param[in] gradients Two gradient images, the first contains the x gradients and the second the y gradients.
	 * @param[out] filtered Image with one channel per gradient, ordered by original channel (x1, y1, x2, y2, and so on).
	 */
	void mergeGradients(const std::vector<cv::Mat>& gradients, cv::Mat& filtered) const;

	/**
	 * Determines the offset of the gradient values.
	 *
	 * @param[in] imageDepth The image depth.
	 * @return The value that is added to the gradient values before storing them.
	 */
	double getDelta(int imageDepth) const;

	int kernelSize; ///< The size of the derivative kernel. Must be positive and odd or CV_SCHARR.
	double kernelScale; ///< Normalization factor of the kernel.
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_GRADIENTFILTER_HPP_ */
