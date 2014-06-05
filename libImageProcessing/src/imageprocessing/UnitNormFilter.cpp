/*
 * UnitNormFilter.cpp
 *
 *  Created on: 02.07.2013
 *      Author: Patrik Huber
 */

#include "imageprocessing/UnitNormFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;

namespace imageprocessing {

const float UnitNormFilter::eps = 1e-4;

UnitNormFilter::UnitNormFilter(int normType) : normType(normType) {}

Mat UnitNormFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.channels() > 1)
		throw invalid_argument("UnitNormFilter: The image must have exactly one channel.");
	image.convertTo(filtered, CV_32F);
	normalize(filtered);
	return filtered;
}

void UnitNormFilter::applyInPlace(Mat& image) const {
	if (image.type() == CV_32FC1)
		normalize(image);
	else
		image = applyTo(image);
}

void UnitNormFilter::normalize(Mat& image) const {
	double norm = cv::norm(image, normType);
	image = image / (norm + eps);
}

} /* namespace imageprocessing */
