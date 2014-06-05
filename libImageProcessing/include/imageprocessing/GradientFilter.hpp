/*
 * GradientFilter.hpp
 *
 *  Created on: 28.05.2013
 *      Author: poschmann
 */

#ifndef GRADIENTFILTER_HPP_
#define GRADIENTFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that computes the gradients of each pixel. The resulting image has two channels, one for the gradient
 * of x and one for the gradient of y. The filtered image data will have the same type as the input image data. In
 * case of an unsigned data type, the neutral value will be 127 (CV_8U) or 65535 (CV_16U) instead of 0.
 */
class GradientFilter : public ImageFilter {
public:

	/**
	 * Constructs a new gradient filter.
	 *
	 * @param[in] kernelSize The size of the Sobel kernel. Must be 1, 3, 5, 7 or CV_SCHARR.
	 * @param[in] blurKernelSize The size of the blurring kernel for additional smoothing. If 0, no additional smoothing will be applied.
	 */
	explicit GradientFilter(int kernelSize, int blurKernelSize = 0);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	void applyInPlace(cv::Mat& image) const;

private:

	/**
	 * Determines the scale factor of the kernel that is used to normalize the values.
	 *
	 * @param[in] kernelSize The size of the Sobel kernel.
	 * @return The normalization factor of the kernel values.
	 */
	double getScale(int kernelSize) const;

	/**
	 * Determines the offset of the gradient values.
	 *
	 * @param[in] imageType The type of the output image.
	 * @return The offset.
	 */
	double getDelta(int imageType) const;

	int kernelSize;     ///< The size of the Sobel kernel. Must be 1, 3, 5, 7 or CV_SCHARR.
	int blurKernelSize; ///< The size of the blurring kernel for additional smoothing.
};

} /* namespace imageprocessing */
#endif /* GRADIENTFILTER_HPP_ */
