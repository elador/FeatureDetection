/*
 * IntegralChannelHistogramFilter.cpp
 *
 *  Created on: 06.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/IntegralChannelHistogramFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;

namespace imageprocessing {

IntegralChannelHistogramFilter::IntegralChannelHistogramFilter(
		unsigned int blockRows, unsigned int blockColumns, double overlap, Normalization normalization) :
				HistogramFilter(normalization), blockRows(blockRows), blockColumns(blockColumns), overlap(overlap) {
	if (blockRows == 0 || blockColumns == 0)
		throw invalid_argument("IntegralChannelHistogramFilter: the amount of rows and columns must be greater than zero");
	if (overlap < 0 || overlap >= 1)
		throw invalid_argument("IntegralChannelHistogramFilter: the overlap must be between zero (inclusive) and one (exclusive)");
}

Mat IntegralChannelHistogramFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.depth() != CV_32S)
		throw invalid_argument("IntegralChannelHistogramFilter: the image must have a depth of CV_32S");

	unsigned int bins = image.channels();
	double shift = 1.0 - overlap;
	double blockWidth = (image.cols - 1) / (1 + (blockColumns - 1) * shift);
	double blockHeight = (image.rows - 1) / (1 + (blockRows - 1) * shift);
	filtered.create(blockRows, blockColumns, CV_32FC(bins));
	float* histogramValues = filtered.ptr<float>();
	float factor = 1.f / 255.f;
	for (unsigned int i = 0; i < blockRows; ++i) {
		for (unsigned int j = 0; j < blockColumns; ++j) {
			int top = cvRound(i * shift * blockHeight);
			int bottom = cvRound(i * shift * blockHeight + blockHeight);
			int left = cvRound(j * shift * blockWidth);
			int right = cvRound(j * shift * blockWidth + blockWidth);
			int* tl = reinterpret_cast<int*>(image.data + top * image.step[0] + left * image.step[1]);
			int* tr = reinterpret_cast<int*>(image.data + top * image.step[0] + right * image.step[1]);
			int* bl = reinterpret_cast<int*>(image.data + bottom * image.step[0] + left * image.step[1]);
			int* br = reinterpret_cast<int*>(image.data + bottom * image.step[0] + right * image.step[1]);
			float l1norm = eps;
			float l2normSquared = eps * eps;
			for (unsigned int b = 0; b < bins; ++b) {
				histogramValues[b] = factor * (tl[b] + br[b] - tr[b] - bl[b]);
				l1norm += histogramValues[b];
				l2normSquared += histogramValues[b] * histogramValues[b];
			}
			if (normalization == Normalization::L2HYS) {
				float normalizer = 1.f / sqrt(l2normSquared);
				l2normSquared = eps * eps;
				for (unsigned int b = 0; b < bins; ++b) {
					histogramValues[b] = std::min(0.2f, normalizer * histogramValues[b]);
					l2normSquared += histogramValues[b] * histogramValues[b];
				}
				normalizer = 1.f / sqrt(l2normSquared);
				for (unsigned int b = 0; b < bins; ++b)
					histogramValues[b] *= normalizer;
			} else {
				float normalizer = 1;
				if (normalization == Normalization::L2NORM)
					normalizer = 1.f / sqrt(l2normSquared);
				else // normalization == Normalization::L1NORM || normalization == Normalization::L1SQRT
					normalizer = 1.f / l1norm;
				if (normalization == Normalization::L1SQRT) {
					for (unsigned int b = 0; b < bins; ++b)
						histogramValues[b] = sqrt(normalizer * histogramValues[b]);
				} else {
					for (unsigned int b = 0; b < bins; ++b)
						histogramValues[b] *= normalizer;
				}
			}
			histogramValues += bins;
		}
	}
	return filtered;
}

} /* namespace imageprocessing */
