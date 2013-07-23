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

GradientHistogramFilter::GradientHistogramFilter(unsigned int bins, bool signedGradients) : bins(bins) {
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	Vec4b binCode;
	// build the look-up table
	// index of the look-up table is the binary concatanation of the gradients of x and y
	// value of the look-up table is the binary concatanation of the bin index and weight (scaled to 255)
	double min = 255, max = -255, maxMag, maxBin, maxGradX, maxGradY;
	gradientCode.gradient.x = 0;
	for (int x = 0; x < 256; ++x) {
		double gradientX = (static_cast<double>(x) - 127) / 127;
		gradientCode.gradient.y = 0;
		for (int y = 0; y < 256; ++y) {
			double gradientY = (static_cast<double>(y) - 127) / 127;
			double direction = atan2(gradientY, gradientX);
			double magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);
			double bin;
			if (signedGradients) {
				direction += CV_PI;
				bin = direction * bins / (2 * CV_PI);
			} else { // unsigned gradients
				if (direction < 0)
					direction += CV_PI;
				bin = direction * bins / CV_PI;
			}
			binCode[0] = static_cast<uchar>(floor(bin)) % bins;
			binCode[2] = static_cast<uchar>(ceil(bin)) % bins;
			binCode[3] = cv::saturate_cast<uchar>(255 * magnitude * (bin - floor(bin)));
			binCode[1] = cv::saturate_cast<uchar>(255 * magnitude - binCode[3]);
			binCodes[gradientCode.index] = binCode;
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

GradientHistogramFilter::~GradientHistogramFilter() {}

unsigned int GradientHistogramFilter::getBinCount() const {
	return bins;
}

Mat GradientHistogramFilter::applyTo(const Mat& image, Mat& filtered) {
	if (image.type() != CV_8UC2)
		throw invalid_argument("GradientHistogramFilter: the image must by of type CV_8UC2");

	int rows = image.rows;
	int cols = image.cols;
	filtered.create(rows, cols, CV_8UC4);
	if (image.isContinuous() && filtered.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	for (int row = 0; row < rows; ++row) {
		const ushort* gradientCode = image.ptr<ushort>(row); // concatenation of x gradient and y gradient (both uchar)
		const cv::Vec2b* gradientCode2 = image.ptr<cv::Vec2b>(row); // concatenation of x gradient and y gradient (both uchar)
		Vec4b* binCode = filtered.ptr<Vec4b>(row); // concatenation of two bin indices and weights (all uchar)
		for (int col = 0; col < cols; ++col)
			binCode[col] = binCodes[gradientCode[col]];
	}
	return filtered;
}

void GradientHistogramFilter::applyInPlace(Mat& image) {
	image = applyTo(image);
}

} /* namespace imageprocessing */
