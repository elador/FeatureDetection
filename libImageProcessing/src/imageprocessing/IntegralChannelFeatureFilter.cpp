/*
 * IntegralChannelFeatureFilter.cpp
 *
 *  Created on: 08.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/IntegralChannelFeatureFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;

namespace imageprocessing {

IntegralChannelFeatureFilter::IntegralChannelFeatureFilter(unsigned int blockRows, unsigned int blockColumns, double overlap) :
		blockRows(blockRows), blockColumns(blockColumns), overlap(overlap) {
	if (blockRows == 0 || blockColumns == 0)
		throw invalid_argument("IntegralChannelFeatureFilter: the amount of rows and columns must be greater than zero");
	if (overlap < 0 || overlap >= 1)
		throw invalid_argument("IntegralChannelFeatureFilter: the overlap must be between zero (inclusive) and one (exclusive)");
}

Mat IntegralChannelFeatureFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.depth() != CV_32S)
		throw invalid_argument("IntegralChannelFeatureFilter: the image must have a depth of CV_32S");

	unsigned int channels = image.channels();
	double shift = 1.0 - overlap;
	double blockWidth = (image.cols - 1) / (1 + (blockColumns - 1) * shift);
	double blockHeight = (image.rows - 1) / (1 + (blockRows - 1) * shift);
	filtered.create(1, blockRows * blockColumns * channels, CV_8U);
	uchar* values = filtered.ptr<uchar>();
	for (unsigned int i = 0; i < blockRows; ++i) {
		for (unsigned int j = 0; j < blockColumns; ++j) {
			int top = cvRound(i * shift * blockHeight);
			int bottom = cvRound(i * shift * blockHeight + blockHeight);
			int left = cvRound(j * shift * blockWidth);
			int right = cvRound(j * shift * blockWidth + blockWidth);
			int area = (bottom - top) * (right - left);
			int* tl = reinterpret_cast<int*>(image.data + top * image.step[0] + left * image.step[1]);
			int* tr = reinterpret_cast<int*>(image.data + top * image.step[0] + right * image.step[1]);
			int* bl = reinterpret_cast<int*>(image.data + bottom * image.step[0] + left * image.step[1]);
			int* br = reinterpret_cast<int*>(image.data + bottom * image.step[0] + right * image.step[1]);
			for (unsigned int b = 0; b < channels; ++b)
				values[b] = (tl[b] + br[b] - tr[b] - bl[b]) / area;
			values += channels;
		}
	}
	return filtered;
}

} /* namespace imageprocessing */
