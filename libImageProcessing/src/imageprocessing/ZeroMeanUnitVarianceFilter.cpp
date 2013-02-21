/*
 * ZeroMeanUnitVarianceFilter.cpp
 *
 *  Created on: 21.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/ZeroMeanUnitVarianceFilter.hpp"

using cv::Scalar;
using cv::meanStdDev;

namespace imageprocessing {

ZeroMeanUnitVarianceFilter::ZeroMeanUnitVarianceFilter() {}

ZeroMeanUnitVarianceFilter::~ZeroMeanUnitVarianceFilter() {}

Mat ZeroMeanUnitVarianceFilter::applyTo(const Mat& image, Mat& filtered) {
	if (image.channels() > 1)
		throw "ZeroMeanUnitVarianceFilter: the image must have exactly one channel";
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
