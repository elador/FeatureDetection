/*
 * HistogramFeatureExtractor.cpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/HistogramFeatureExtractor.hpp"

namespace imageprocessing {

HistogramFeatureExtractor::HistogramFeatureExtractor(Normalization normalization) : normalization(normalization) {}

HistogramFeatureExtractor::~HistogramFeatureExtractor() {}

void HistogramFeatureExtractor::normalize(Mat& histogram) const {
	switch (normalization) {
		case Normalization::L2NORM: normalizeL2(histogram); break;
		case Normalization::L2HYS:  normalizeL2Hys(histogram); break;
		case Normalization::L1NORM: normalizeL1(histogram); break;
		case Normalization::L1SQRT: normalizeL1Sqrt(histogram); break;
		case Normalization::NONE:   break;
	}
}

const float HistogramFeatureExtractor::eps = 1e-3;

void HistogramFeatureExtractor::normalizeL2(Mat& histogram) const {
	float normSquared = cv::norm(histogram, cv::NORM_L2SQR);
	histogram = histogram / sqrt(normSquared + eps * eps);
}

void HistogramFeatureExtractor::normalizeL2Hys(Mat& histogram) const {
	normalizeL2(histogram);
	histogram = min(histogram, 0.2);
	normalizeL2(histogram);
}

void HistogramFeatureExtractor::normalizeL1(Mat& histogram) const {
	float norm = cv::norm(histogram, cv::NORM_L1);
	histogram = histogram / (norm + eps);
}

void HistogramFeatureExtractor::normalizeL1Sqrt(Mat& histogram) const {
	normalizeL1(histogram);
	cv::sqrt(histogram, histogram);
}

} /* namespace imageprocessing */
