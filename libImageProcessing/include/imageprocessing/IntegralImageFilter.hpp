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
 * Image filter that transforms the image to an integral image.
 *
 * The output image will have the same size as the input image and be of type CV_32F. The row and columns of
 * zeroes are not contained within the integral image.
 */
class IntegralImageFilter : public ImageFilter {
public:

	/**
	 * Constructs a new integral image filter.
	 *
	 * @param[in] squared Flag that indicates whether the pixel values should be squared before summing up.
	 */
	IntegralImageFilter(bool squared);

	~IntegralImageFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered);

	void applyInPlace(Mat& image);

private:

	bool squared; ///< Flag that indicates whether the pixel values should be squared before summing up.
};

} /* namespace imageprocessing */
#endif /* INTEGRALIMAGEFILTER_HPP_ */
