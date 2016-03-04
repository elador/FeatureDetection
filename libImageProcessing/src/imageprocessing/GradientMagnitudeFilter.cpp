/*
 * GradientMagnitudeFilter.cpp
 *
 *  Created on: 05.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/GradientMagnitudeFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Vec2b;
using std::invalid_argument;

namespace imageprocessing {

GradientMagnitudeFilter::GradientMagnitudeFilter() {}

Mat GradientMagnitudeFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.channels() != 2)
		throw invalid_argument("GradientMagnitudeFilter: the image must have two channels");

	filtered.create(image.rows, image.cols, image.depth());
	int rows = image.rows;
	int cols = image.cols;
	if (image.isContinuous() && filtered.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	switch (image.depth()) {
		case CV_8U: computeMagnitude<uchar>(image, filtered, rows, cols, 127); break;
		case CV_8S: computeMagnitude<char>(image, filtered, rows, cols); break;
		case CV_16U: computeMagnitude<ushort>(image, filtered, rows, cols, 65535); break;
		case CV_16S: computeMagnitude<short>(image, filtered, rows, cols); break;
		case CV_32S: computeMagnitude<int>(image, filtered, rows, cols); break;
		case CV_32F: computeMagnitude<float>(image, filtered, rows, cols); break;
		case CV_64F: computeMagnitude<double>(image, filtered, rows, cols); break;
		default: throw invalid_argument("GradientMagnitudeFilter: unsupported image depth " + std::to_string(image.depth()));
	}

	return filtered;
}

} /* namespace imageprocessing */
