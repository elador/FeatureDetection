/*
 * GradientBinningFilter.cpp
 *
 *  Created on: 28.05.2013
 *      Author: poschmann
 */

#include "imageprocessing/GradientBinningFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Vec2b;
using cv::Vec4b;
using std::invalid_argument;

namespace imageprocessing {

GradientBinningFilter::GradientBinningFilter(unsigned int bins, bool signedGradients, bool interpolate) :
		bins(bins), interpolate(interpolate), resultType(interpolate ? CV_8UC4 : CV_8UC2) {
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	Vec2b oneBinCode;
	Vec4b twoBinCode;
	// build the look-up table
	// index of the look-up table is the binary concatanation of the gradients of x and y
	// value of the look-up table is the binary concatanation of two bin indices and weights (scaled to 255)
	gradientCode.gradient.x = 0;
	for (int x = 0; x < 256; ++x) {
		double gradientX = (static_cast<double>(x) - 127) / 255;
		gradientCode.gradient.y = 0;
		for (int y = 0; y < 256; ++y) {
			double gradientY = (static_cast<double>(y) - 127) / 255;
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
			oneBinCode[0] = static_cast<uchar>(round(bin)) % bins;
			oneBinCode[1] = cv::saturate_cast<uchar>(255 * magnitude);
			oneBinCodes[gradientCode.index] = oneBinCode;
			twoBinCode[0] = static_cast<uchar>(floor(bin)) % bins;
			twoBinCode[2] = static_cast<uchar>(ceil(bin)) % bins;
			twoBinCode[3] = cv::saturate_cast<uchar>(255 * magnitude * (bin - floor(bin)));
			twoBinCode[1] = cv::saturate_cast<uchar>(255 * magnitude - twoBinCode[3]);
			twoBinCodes[gradientCode.index] = twoBinCode;
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

unsigned int GradientBinningFilter::getBinCount() const {
	return bins;
}

Mat GradientBinningFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.type() != CV_8UC2)
		throw invalid_argument("GradientHistogramFilter: the image must be of type CV_8UC2");

	int rows = image.rows;
	int cols = image.cols;
	filtered.create(rows, cols, resultType);
	if (image.isContinuous() && filtered.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	if (interpolate) {
		for (int row = 0; row < rows; ++row) {
			const ushort* gradientCode = image.ptr<ushort>(row); // concatenation of x gradient and y gradient (both uchar)
			Vec4b* binCode = filtered.ptr<Vec4b>(row); // concatenation of two bin indices and weights (all uchar)
			for (int col = 0; col < cols; ++col)
				binCode[col] = twoBinCodes[gradientCode[col]];
		}
	} else {
		for (int row = 0; row < rows; ++row) {
			const ushort* gradientCode = image.ptr<ushort>(row); // concatenation of x gradient and y gradient (both uchar)
			Vec2b* binCode = filtered.ptr<Vec2b>(row); // concatenation of bin index and weight (both uchar)
			for (int col = 0; col < cols; ++col)
				binCode[col] = oneBinCodes[gradientCode[col]];
		}
	}
	return filtered;
}

} /* namespace imageprocessing */
