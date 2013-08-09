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

	/**
	 * Constructs a new image filter.
	 */
	ImageFilter() {}

	virtual ~ImageFilter() {}

	/**
	 * Applies this filter to an image, writing the filtered image data into a newly created image.
	 *
	 * @param[in] image The image that should be filtered.
	 * @return The filtered image.
	 */
	Mat applyTo(const Mat& image) const {
		Mat filtered;
		applyTo(image, filtered);
		return filtered;
	}

	/**
	 * Applies this filter to an image, writing the filtered image data into a new image.
	 *
	 * @param[in] image The image that should be filtered.
	 * @param[out] filtered The image for writing the filtered data into.
	 * @return The filtered image.
	 */
	virtual Mat applyTo(const Mat& image, Mat& filtered) const = 0;

	/**
	 * Applies this filter to an image, writing the filtered data into the image itself.
	 *
	 * @param[in,out] image The image that should be filtered.
	 */
	virtual void applyInPlace(Mat& image) const {
		image = applyTo(image);
	}
};

} /* namespace imageprocessing */
#endif /* IMAGEFILTER_HPP_ */
