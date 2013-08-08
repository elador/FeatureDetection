/*
 * HistogramFilter.cpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/HistogramFilter.hpp"

namespace imageprocessing {

HistogramFilter::HistogramFilter(Normalization normalization) : normalization(normalization) {}

HistogramFilter::~HistogramFilter() {}

void HistogramFilter::applyInPlace(Mat& image) const {
	image = applyTo(image);
}

void HistogramFilter::normalize(Mat& histogram) const {
	switch (normalization) {
		case Normalization::L2NORM: normalizeL2(histogram); break;
		case Normalization::L2HYS:  normalizeL2Hys(histogram); break;
		case Normalization::L1NORM: normalizeL1(histogram); break;
		case Normalization::L1SQRT: normalizeL1Sqrt(histogram); break;
		case Normalization::NONE:   break;
	}
}

const float HistogramFilter::eps = 1e-4;

void HistogramFilter::normalizeL2(Mat& histogram) const {
	float normSquared = cv::norm(histogram, cv::NORM_L2SQR);
	histogram = histogram / sqrt(normSquared + eps * eps);
}

void HistogramFilter::normalizeL2Hys(Mat& histogram) const {
	normalizeL2(histogram);
	histogram = min(histogram, 0.2);
	normalizeL2(histogram);
}

void HistogramFilter::normalizeL1(Mat& histogram) const {
	float norm = cv::norm(histogram, cv::NORM_L1);
	histogram = histogram / (norm + eps);
}

void HistogramFilter::normalizeL1Sqrt(Mat& histogram) const {
	normalizeL1(histogram);
	cv::sqrt(histogram, histogram);
}

} /* namespace imageprocessing */
