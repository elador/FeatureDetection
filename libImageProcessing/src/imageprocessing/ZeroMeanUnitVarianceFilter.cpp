/*
 * ZeroMeanUnitVarianceFilter.cpp
 *
 *  Created on: 21.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/ZeroMeanUnitVarianceFilter.hpp"
#include <stdexcept>

using cv::Scalar;
using cv::meanStdDev;
using std::invalid_argument;

namespace imageprocessing {

ZeroMeanUnitVarianceFilter::ZeroMeanUnitVarianceFilter() {}

ZeroMeanUnitVarianceFilter::~ZeroMeanUnitVarianceFilter() {}

Mat ZeroMeanUnitVarianceFilter::applyTo(const Mat& image, Mat& filtered) {
	if (image.channels() > 1)
		throw invalid_argument("ZeroMeanUnitVarianceFilter: the image must have exactly one channel");
	Scalar mean, deviation;
	meanStdDev(image, mean, deviation);
	image.convertTo(filtered, CV_32F);
	filtered = (filtered - mean) / deviation[0];
	return filtered;
}

void ZeroMeanUnitVarianceFilter::applyInPlace(Mat& image) {
	image = applyTo(image);
}

} /* namespace imageprocessing */
