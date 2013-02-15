/*
 * ImageFilter.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#ifndef IMAGEFILTER_HPP_
#define IMAGEFILTER_HPP_

#include "opencv2/core/core.hpp"

using cv::Mat;

namespace imageprocessing {

/**
 * Filter of images.
 */
class ImageFilter {
public:

	virtual ~ImageFilter() {}

	/**
	 * Creates a new filtered image.
	 *
	 * @param[in] in The input image.
	 * @return The filtered image.
	 */
	Mat filter(const Mat &in);

	/**
	 * Writes the filtered image data into the output image.
	 *
	 * @param[in] in The input image.
	 * @param[in] out The output image.
	 */
	virtual void filter(const Mat &in, Mat &out) = 0;
};

} /* namespace imageprocessing */
#endif /* IMAGEFILTER_HPP_ */
