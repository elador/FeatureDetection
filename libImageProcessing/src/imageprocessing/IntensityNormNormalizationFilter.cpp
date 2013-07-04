/*
 * IntensityNormNormalizationFilter.cpp
 *
 *  Created on: 02.07.2013
 *      Author: Patrik Huber
 */

#include "imageprocessing/IntensityNormNormalizationFilter.hpp"
#include <stdexcept>

using cv::Scalar;
using std::invalid_argument;

namespace imageprocessing {

IntensityNormNormalizationFilter::IntensityNormNormalizationFilter(int normType) : normType(normType) {}

IntensityNormNormalizationFilter::~IntensityNormNormalizationFilter() {}

Mat IntensityNormNormalizationFilter::applyTo(const Mat& image, Mat& filtered) {
	if (image.channels() > 1)
		throw invalid_argument("IntensityNormNormalizationFilter: The image must have exactly one channel.");
	
	image.convertTo(filtered, CV_32F);
	double norm = cv::norm(image, normType);
	filtered = filtered / norm;

	return filtered;
}

void IntensityNormNormalizationFilter::applyInPlace(Mat& image) {
	image = applyTo(image);
}

} /* namespace imageprocessing */
