/*
 * ColorSpaceConversionFilter.cpp
 *
 *  Created on: 08.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/ColorSpaceConversionFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using cv::Mat;

namespace imageprocessing {

ColorSpaceConversionFilter::ColorSpaceConversionFilter(int conversion) : conversion(conversion) {}

Mat ColorSpaceConversionFilter::applyTo(const Mat& image, Mat& filtered) const {
	cv::cvtColor(image, filtered, conversion);
	return filtered;
}

} /* namespace imageprocessing */
