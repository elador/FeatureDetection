/*
 * UnitNormFilter.hpp
 *
 *  Created on: 02.07.2013
 *      Author: Patrik Huber
 */

#ifndef UNITNORMFILTER_HPP_
#define UNITNORMFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that normalizes the pixel values by the norm of the whole
 * image. The norm can be any of OpenCV's norm constants, like NORM_L2 or
 * NORM_L1.
 * The input image must not have more than one channel and can be of any type. The
 * filtered image is of type CV_32F with the values in the same range.
 */
class UnitNormFilter : public ImageFilter {
public:

	/**
	 * Constructs a new filter that normalizes an image by its intensity-norm.
	 *
	 * @param[in] normType The norm type (see cv::norm).
	 */
	UnitNormFilter(int normType = cv::NORM_L2);

	~UnitNormFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered) const;

	void applyInPlace(Mat& image) const;

private:

	int normType; ///< The norm type (see cv::norm).
};

} /* namespace imageprocessing */
#endif /* UNITNORMFILTER_HPP_ */
