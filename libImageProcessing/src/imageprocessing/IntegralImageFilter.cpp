/*
 * IntegralImageFilter.cpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/IntegralImageFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using cv::Mat;

namespace imageprocessing {

IntegralImageFilter::IntegralImageFilter(int type) : type(type) {}

Mat IntegralImageFilter::applyTo(const Mat& image, Mat& filtered) const {
	cv::integral(image, filtered, type);
	return filtered;
}

} /* namespace imageprocessing */
