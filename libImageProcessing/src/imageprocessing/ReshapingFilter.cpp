/*
 * ReshapingFilter.cpp
 *
 *  Created on: 25.04.2013
 *      Author: poschmann
 */

#include "imageprocessing/ReshapingFilter.hpp"

using cv::Mat;

namespace imageprocessing {

ReshapingFilter::ReshapingFilter(int rows, int channels) : rows(rows), channels(channels) {}

Mat ReshapingFilter::applyTo(const Mat& image, Mat& filtered) const {
	image.copyTo(filtered);
	filtered = filtered.reshape(channels, rows);
	return filtered;
}

void ReshapingFilter::applyInPlace(Mat& image) const {
	applyTo(image, image);
}

} /* namespace imageprocessing */
