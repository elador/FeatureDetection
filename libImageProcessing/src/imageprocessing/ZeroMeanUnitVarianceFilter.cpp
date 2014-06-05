/*
 * ZeroMeanUnitVarianceFilter.cpp
 *
 *  Created on: 21.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/ZeroMeanUnitVarianceFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Scalar;
using cv::meanStdDev;
using std::invalid_argument;

namespace imageprocessing {

ZeroMeanUnitVarianceFilter::ZeroMeanUnitVarianceFilter() {}

Mat ZeroMeanUnitVarianceFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.channels() > 1)
		throw invalid_argument("ZeroMeanUnitVarianceFilter: The image must have exactly one channel.");
	Scalar mean, deviation;
	meanStdDev(image, mean, deviation);
	image.convertTo(filtered, CV_32F);

	if (deviation[0] == 0)
		filtered = Scalar(0);
	else
		filtered = (filtered - mean) / deviation[0];
	return filtered;
}

} /* namespace imageprocessing */
