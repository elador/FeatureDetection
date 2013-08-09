/*
 * UnitNormFilter.cpp
 *
 *  Created on: 02.07.2013
 *      Author: Patrik Huber
 */

#include "imageprocessing/UnitNormFilter.hpp"
#include <stdexcept>

using cv::Scalar;
using std::invalid_argument;

namespace imageprocessing {

UnitNormFilter::UnitNormFilter(int normType) : normType(normType) {}

UnitNormFilter::~UnitNormFilter() {}

Mat UnitNormFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.channels() > 1)
		throw invalid_argument("UnitNormFilter: The image must have exactly one channel.");
	
	image.convertTo(filtered, CV_32F);
	double norm = cv::norm(image, normType);
	filtered = filtered / norm;

	return filtered;
}

} /* namespace imageprocessing */
