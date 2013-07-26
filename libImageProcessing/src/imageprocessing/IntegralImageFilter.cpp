/*
 * IntegralImageFilter.cpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/IntegralImageFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace imageprocessing {

IntegralImageFilter::IntegralImageFilter(int type) : type(type) {}

IntegralImageFilter::~IntegralImageFilter() {}

Mat IntegralImageFilter::applyTo(const Mat& image, Mat& filtered) const {
	cv::integral(image, filtered, type);
	return filtered;
}

void IntegralImageFilter::applyInPlace(Mat& image) const {
	image = applyTo(image);
}

} /* namespace imageprocessing */
