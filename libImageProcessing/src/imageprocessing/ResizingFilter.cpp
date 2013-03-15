/*
 * ResizingFilter.cpp
 *
 *  Created on: 15.03.2013
 *      Author: poschmann
 */

#include "imageprocessing/ResizingFilter.hpp"

using cv::resize;

namespace imageprocessing {

ResizingFilter::ResizingFilter(Size size, int interpolation) : size(size), interpolation(interpolation) {}

ResizingFilter::~ResizingFilter() {}

Mat ResizingFilter::applyTo(const Mat& image, Mat& filtered) {
	resize(image, filtered, size, 0, 0, interpolation);
	return filtered;
}

void ResizingFilter::applyInPlace(Mat& image) {
	image = applyTo(image);
}

} /* namespace imageprocessing */
