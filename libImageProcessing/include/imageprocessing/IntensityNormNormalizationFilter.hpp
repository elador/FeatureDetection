/*
 * IntensityNormNormalizationFilter.hpp
 *
 *  Created on: 02.07.2013
 *      Author: Patrik Huber
 */

#ifndef INTENSITYNORMNORMALIZATIONFILTER_HPP_
#define INTENSITYNORMNORMALIZATIONFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that normalizes the pixel values by the norm of the whole
 * image. The norm can be any of OpenCV's norm constants, like NORM_L2 or
 * NORM_L1.
 * The input image must not have more than one channel and can be of any type. The
 * filtered image is of type CV_32F with the values in the same range.
 */
class IntensityNormNormalizationFilter : public ImageFilter {
public:

	/**
	 * Constructs a new filter that normalizes an image by its intensity-norm.
	 *
	 * @param[in] normType The type of norm to use.
	 */
	IntensityNormNormalizationFilter(int normType = cv::NORM_L2);

	~IntensityNormNormalizationFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered);

	void applyInPlace(Mat& image);

private:
	int normType;
};

} /* namespace imageprocessing */
#endif /* INTENSITYNORMNORMALIZATIONFILTER_HPP_ */
