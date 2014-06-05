/*
 * GradientSumFilter.hpp
 *
 *  Created on: 15.08.2013
 *      Author: poschmann
 */

#ifndef GRADIENTSUMFILTER_HPP_
#define GRADIENTSUMFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Filter that expects a two-channel gradient image and creates a vector containing the sums over the gradient values
 * and absolute gradient values. Similar to SURF, the values and absolute values of the x- and y-gradients are summed
 * up over cells, so each cell contributes four values to the resulting vector. The input image has to be of type
 * CV_8UC2, the result will be a row-vector of type CV_32FC1.
 */
class GradientSumFilter : public ImageFilter {
public:

	/**
	 * Constructs a new gradient sum filter.
	 *
	 * @param[in] rows Row count of cells over which to sum up the gradients.
	 * @param[in] cols Column count of cells over which to sum up the gradients.
	 */
	GradientSumFilter(int rows, int cols);

	/**
	 * Constructs a new gradient sum filter with square cells.
	 *
	 * @param[in] count Row and column count of cells over which to sum up the gradients.
	 */
	explicit GradientSumFilter(int count);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	int rows; ///< Row count of cells over which to sum up the gradients.
	int cols; ///< Column count of cells over which to sum up the gradients.
};

} /* namespace imageprocessing */
#endif /* GRADIENTSUMFILTER_HPP_ */
