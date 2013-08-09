/*
 * ColorSpaceConversionFilter.hpp
 *
 *  Created on: 08.08.2013
 *      Author: poschmann
 */

#ifndef COLORSPACECONVERSIONFILTER_HPP_
#define COLORSPACECONVERSIONFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that converts images from one color space to another.
 */
class ColorSpaceConversionFilter : public ImageFilter {
public:

	/**
	 * Constructs a new color space conversion filter.
	 *
	 * @param[in] conversion The conversion code, see cv::cvtColor for details.
	 */
	explicit ColorSpaceConversionFilter(int conversion);

	~ColorSpaceConversionFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered) const;

private:

	int conversion; ///< The conversion code, see cv::cvtColor for details.
};

} /* namespace imageprocessing */
#endif /* COLORSPACECONVERSIONFILTER_HPP_ */
