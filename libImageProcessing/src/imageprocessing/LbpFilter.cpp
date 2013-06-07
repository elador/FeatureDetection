/*
 * LbpFilter.cpp
 *
 *  Created on: 05.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/LbpFilter.hpp"
#include <stdexcept>

using cv::Ptr;
using cv::BaseRowFilter;
using cv::BaseColumnFilter;
using cv::FilterEngine;
using std::invalid_argument;

namespace imageprocessing {

LbpFilter::LbpFilter(bool uniform) : uniform(uniform) {
	if (uniform) {
		int nonUniformIndex = 0;
		int emptyIndex = 1;
		for (unsigned int i = 0; i < map.size(); ++i) {
			if (isUniform(static_cast<uchar>(i)))
				map[i] = emptyIndex++;
			else
				map[i] = nonUniformIndex;
		}
	}
}

LbpFilter::~LbpFilter() {}

bool LbpFilter::isUniform(uchar code) {
	int transitions = 0;
	int previousBit = (code >> 7) & 1;
	for (int pos = 0; pos < 8; ++pos) {
		int currentBit = (code >> pos) & 1;
		if (previousBit != currentBit) {
			transitions++;
			previousBit = currentBit;
		}
	}
	return transitions <= 2;
}

Mat LbpFilter::applyTo(const Mat& image, Mat& filtered) {
	if (image.channels() > 1)
		throw invalid_argument("LbpFilter: the image must have exactly one channel");
	Ptr<BaseFilter> filter2D;
	switch (image.type()) {
		case CV_8U:  filter2D = Ptr<BaseFilter>(new Filter<uchar>()); break;
		case CV_8S:  filter2D = Ptr<BaseFilter>(new Filter<char>()); break;
		case CV_16U: filter2D = Ptr<BaseFilter>(new Filter<ushort>()); break;
		case CV_16S: filter2D = Ptr<BaseFilter>(new Filter<short>()); break;
		case CV_32S: filter2D = Ptr<BaseFilter>(new Filter<int>()); break;
		case CV_32F: filter2D = Ptr<BaseFilter>(new Filter<float>()); break;
		case CV_64F: filter2D = Ptr<BaseFilter>(new Filter<double>()); break;
		default: throw invalid_argument("LbpFilter: unsupported image type " + image.type());
	}
	filtered.create(image.rows, image.cols, CV_8U);
	Ptr<FilterEngine> filterEngine(new FilterEngine(filter2D, Ptr<BaseRowFilter>(), Ptr<BaseColumnFilter>(),
			image.type(), CV_8U, image.type(), cv::BORDER_REPLICATE));
	filterEngine->apply(image, filtered);

	if (uniform) {
		int rows = filtered.rows;
		int cols = filtered.cols;
		if (filtered.isContinuous()) {
			cols *= rows;
			rows = 1;
		}
		for (int row = 0; row < rows; ++row) {
			uchar* values = filtered.ptr<uchar>(0);
			for (int col = 0; col < cols; ++col)
				values[col] = map[values[col]];
		}
	}
	return filtered;
}

void LbpFilter::applyInPlace(Mat& image) {
	applyTo(image, image);
}

} /* namespace imageprocessing */
