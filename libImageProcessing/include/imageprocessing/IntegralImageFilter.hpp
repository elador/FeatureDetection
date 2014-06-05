/*
 * IntegralImageFilter.hpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#ifndef INTEGRALIMAGEFILTER_HPP_
#define INTEGRALIMAGEFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that transforms the image to an integral image. Each pixel of the resulting integral image contains
 * the sum over all the pixel values above and to the left. Therefore, the first row and first column will contain
 * zeroes and the integral image will by larger than the input image by one row and column.
 *
 * If the desired type of the integral image is negative, then it will be chosen automatically. If the input image
 * is of type CV_8U, it will be CV_32S, otherwise it will be CV_64F (according to the code of OpenCV 2.4.3).
 */
class IntegralImageFilter : public ImageFilter {
public:

	/**
	 * Constructs a new integral image filter.
	 *
	 * @param[in] type The type (depth) of the filtered images. If negative, the type is chosen automatically.
	 */
	explicit IntegralImageFilter(int type = -1);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	int type; ///< The type (depth) of the filtered images. If negative, the type is chosen automatically.
};

} /* namespace imageprocessing */
#endif /* INTEGRALIMAGEFILTER_HPP_ */
