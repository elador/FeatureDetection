/*
 * IntegralGradientFilter.hpp
 *
 *  Created on: 14.08.2013
 *      Author: poschmann
 */

#ifndef INTEGRALGRADIENTFILTER_HPP_
#define INTEGRALGRADIENTFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Filter that expects an integral image and computes horizontal and vertical gradients by taking the differences of
 * neighboring regions across the image. The input image must be of type CV_32SC1, the output image will be of type
 * CV_8UC2.
 */
class IntegralGradientFilter : public ImageFilter {
public:

	/**
	 * Constructs a new integral gradient filter.
	 *
	 * @param[in] rows Row count of points at which to compute the gradients.
	 * @param[in] cols Column count of points at which to compute the gradients.
	 */
	IntegralGradientFilter(int rows, int cols);

	/**
	 * Constructs a new integral gradient filter with square results.
	 *
	 * @param[in] count Row and column count of points at which to compute the gradients.
	 */
	IntegralGradientFilter(int count);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	int rows; ///< Row count of points at which to compute the gradients.
	int cols; ///< Column count of points at which to compute the gradients.
};

} /* namespace imageprocessing */
#endif /* INTEGRALGRADIENTFILTER_HPP_ */
