/*
 * GradientHistogramFilter.cpp
 *
 *  Created on: 28.05.2013
 *      Author: poschmann
 */

#include "imageprocessing/GradientHistogramFilter.hpp"
#include <stdexcept>

using std::invalid_argument;

namespace imageprocessing {

GradientHistogramFilter::GradientHistogramFilter(int bins, bool signedGradients, double offset) : offset(offset) {
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	Vec2b binCode;
	// build the look-up table
	// index of the look-up table is the binary concatanation of the gradients of x and y
	// value of the look-up table is the binary concatanation of the bin index and weight (scaled to 255)
	gradientCode.gradient.x = 0;
	for (int x = 0; x < 256; ++x) {
		double gradientX = (static_cast<double>(x) - 127) / 127;
		gradientCode.gradient.y = 0;
		for (int y = 0; y < 256; ++y) {
			double gradientY = (static_cast<double>(y) - 127) / 127;
			double direction = atan2(gradientY, gradientX);
			double magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);
			if (signedGradients) {
				direction += CV_PI;
				binCode[0] = static_cast<uchar>(floor((direction + offset) * bins / (2 * CV_PI))) % bins;
			} else { // unsigned gradients
				if (direction < 0)
					direction += CV_PI;
				binCode[0] = static_cast<uchar>(floor((direction + offset) * bins / CV_PI)) % bins;
			}
			binCode[1] = cv::saturate_cast<uchar>(255 * magnitude);
			binCodes[gradientCode.index] = binCode;
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

GradientHistogramFilter::~GradientHistogramFilter() {}

Mat GradientHistogramFilter::applyTo(const Mat& image, Mat& filtered) {
	if (image.type() != CV_8UC2)
		throw invalid_argument("GradientHistogramFilter: the image must by of type CV_8UC2");

	int rows = image.rows;
	int cols = image.cols;
	filtered.create(rows, cols, CV_8UC2);
	if (image.isContinuous() && filtered.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	for (int row = 0; row < rows; ++row) {
		const ushort* gradientCode = image.ptr<ushort>(row); // concatenation of x gradient and y gradient (both uchar)
		Vec2b* binCode = filtered.ptr<Vec2b>(row); // concatenation of bin index and weight (both uchar)
		for (int col = 0; col < cols; ++col) {
			binCode[col] = binCodes[gradientCode[col]];
		}
	}
	return filtered;
}

void GradientHistogramFilter::applyInPlace(Mat& image) {
	applyTo(image, image);
}

} /* namespace imageprocessing */
