/*
 * ResizingFilter.hpp
 *
 *  Created on: 15.03.2013
 *      Author: poschmann
 */

#ifndef RESIZINGFILTER_HPP_
#define RESIZINGFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using cv::Size;

namespace imageprocessing {

/**
 * Filter that resizes images to a certain size.
 */
class ResizingFilter : public ImageFilter {
public:

	/**
	 * Constructs a new resizing filter.
	 *
	 * @param[in] size The size of the filtered images.
	 * @param[in] interpolation The interpolation method (see last parameter of cv::resize).
	 */
	explicit ResizingFilter(Size size, int interpolation = cv::INTER_AREA);

	~ResizingFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered) const;

private:

	Size size;         ///< The size of the filtered images.
	int interpolation; ///< The interpolation method.
};

} /* namespace imageprocessing */
#endif /* RESIZINGFILTER_HPP_ */
