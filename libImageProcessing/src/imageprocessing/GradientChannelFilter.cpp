/*
 * GradientChannelFilter.cpp
 *
 *  Created on: 05.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/GradientChannelFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;

namespace imageprocessing {

GradientChannelFilter::GradientChannelFilter(unsigned int bins, bool magnitude, bool signedGradients) :
		bins(bins), magnitude(magnitude) {
	typedef union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} GradientCode;

	BinData binData;
	GradientCode gradientCode;
	// build the look-up table
	// index of the look-up table is the binary concatanation of the gradients of x and y
	// value of the look-up table is the bin information (two bin indices and weights (scaled to 255))
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
			binData.bin1 = static_cast<int>(floor(bin)) % bins;
			binData.bin2 = static_cast<int>(ceil(bin)) % bins;
			binData.weight2 = cv::saturate_cast<uchar>(255 * magnitude * (bin - floor(bin)));
			binData.weight1 = cv::saturate_cast<uchar>(255 * magnitude - binData.weight2);
			gradient2bin[gradientCode.index] = binData;
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

Mat GradientChannelFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.type() != CV_8UC2)
		throw invalid_argument("GradientChannelFilter: the image must be of type CV_8UC2");

	int rows = image.rows;
	int cols = image.cols;
	if (magnitude) { // include magnitude
		filtered.create(rows, cols, CV_8UC(bins + 1));
		if (image.isContinuous() && filtered.isContinuous()) {
			cols *= rows;
			rows = 1;
		}
		for (int row = 0; row < rows; ++row) {
			uchar* values = filtered.data + filtered.step[0] * row;
			const ushort* gradientCode = image.ptr<ushort>(row); // concatenation of x gradient and y gradient (both uchar)
			for (int col = 0; col < cols; ++col) {
				for (unsigned int bin = 0; bin <= bins; ++bin)
					values[bin] = 0;
				BinData binData = gradient2bin[gradientCode[col]];
				values[binData.bin1] += binData.weight1;
				values[binData.bin2] += binData.weight2;
				values[bins] += binData.weight1 + binData.weight2;
				values += bins + 1;
			}
		}
	} else { // do not include magnitude
		filtered.create(rows, cols, CV_8UC(bins));
		if (image.isContinuous() && filtered.isContinuous()) {
			cols *= rows;
			rows = 1;
		}
		for (int row = 0; row < rows; ++row) {
			uchar* values = filtered.data + filtered.step[0] * row;
			const ushort* gradientCode = image.ptr<ushort>(row); // concatenation of x gradient and y gradient (both uchar)
			for (int col = 0; col < cols; ++col) {
				for (unsigned int bin = 0; bin < bins; ++bin)
					values[bin] = 0;
				BinData binData = gradient2bin[gradientCode[col]];
				values[binData.bin1] += binData.weight1;
				values[binData.bin2] += binData.weight2;
				values += bins;
			}
		}
	}

	return filtered;
}

} /* namespace imageprocessing */
