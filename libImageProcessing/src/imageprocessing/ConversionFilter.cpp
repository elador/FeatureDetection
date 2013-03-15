/*
 * ConversionFilter.cpp
 *
 *  Created on: 15.03.2013
 *      Author: poschmann
 */

#include "imageprocessing/ConversionFilter.hpp"

namespace imageprocessing {

ConversionFilter::ConversionFilter(int type, double alpha, double beta) : type(type), alpha(alpha), beta(beta) {}

ConversionFilter::~ConversionFilter() {}

Mat ConversionFilter::applyTo(const Mat& image, Mat& filtered) {
	image.convertTo(filtered, type, alpha, beta);
	return filtered;
}

void ConversionFilter::applyInPlace(Mat& image) {
	applyTo(image, image);
}

} /* namespace imageprocessing */
