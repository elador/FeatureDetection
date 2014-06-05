/*
 * LbpFilter.cpp
 *
 *  Created on: 05.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/LbpFilter.hpp"

using cv::Mat;
using cv::Ptr;
using cv::BaseFilter;
using cv::BaseRowFilter;
using cv::BaseColumnFilter;
using cv::FilterEngine;
using std::invalid_argument;

namespace imageprocessing {

LbpFilter::LbpFilter(Type type) : type(type) {
	if (type == Type::LBP8_UNIFORM) {
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

bool LbpFilter::isUniform(uchar code) const {
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

unsigned int LbpFilter::getBinCount() const {
	switch (type) {
		case Type::LBP8: return 256;
		case Type::LBP8_UNIFORM: return 59;
		case Type::LBP4:
		case Type::LBP4_ROTATED: return 16;
		default: throw std::runtime_error("LbpFilter: invalid type (should never occur)");
	}
}

Mat LbpFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.channels() > 1)
		throw invalid_argument("LbpFilter: the image must have exactly one channel");
	Ptr<BaseFilter> filter2D;
	switch (type) {
		case Type::LBP8: filter2D = createBaseFilter<Lbp8Filter>(image.type()); break;
		case Type::LBP8_UNIFORM: filter2D = createBaseFilter<Lbp8Filter>(image.type()); break;
		case Type::LBP4: filter2D = createBaseFilter<Lbp4Filter>(image.type()); break;
		case Type::LBP4_ROTATED: filter2D = createBaseFilter<RotatedLbp4Filter>(image.type()); break;
	}
	filtered.create(image.rows, image.cols, CV_8U);
	Ptr<FilterEngine> filterEngine(new FilterEngine(filter2D, Ptr<BaseRowFilter>(), Ptr<BaseColumnFilter>(),
			image.type(), CV_8U, image.type(), cv::BORDER_REPLICATE));
	filterEngine->apply(image, filtered);

	if (type == Type::LBP8_UNIFORM) {
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

void LbpFilter::applyInPlace(Mat& image) const {
	applyTo(image, image);
}

} /* namespace imageprocessing */
