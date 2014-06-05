/*
 * ColorBinningFilter.cpp
 *
 *  Created on: 09.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/ColorBinningFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Vec3b;
using cv::Vec4b;
using std::invalid_argument;

namespace imageprocessing {

ColorBinningFilter::ColorBinningFilter(unsigned int bins) : bins(bins) {
	BinData binData;
	for (uchar hue = 0; hue <= 180; ++hue) {
		double bin = static_cast<double>(hue) * static_cast<double>(bins) / 180.0;
		binData.bin1 = static_cast<int>(floor(bin)) % bins;
		binData.bin2 = static_cast<int>(ceil(bin)) % bins;
		binData.weight2 = bin - floor(bin);
		binData.weight1 = 1.f - binData.weight2;
		color2bin[hue] = binData;
	}
}

unsigned int ColorBinningFilter::getBinCount() const {
	return bins;
}

Mat ColorBinningFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.type() != CV_8UC3)
		throw invalid_argument("ColorHistogramFilter: the image must be of type CV_8UC3");

	int rows = image.rows;
	int cols = image.cols;
	filtered.create(rows, cols, CV_8UC4);
	if (image.isContinuous() && filtered.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	float normalizer = 1.f / 255.f;
	for (int row = 0; row < rows; ++row) {
		const Vec3b* colorCode = image.ptr<Vec3b>(row);
		Vec4b* binCodes = filtered.ptr<Vec4b>(row); // concatenation of two bin indices and weights (all uchar)
		for (int col = 0; col < cols; ++col) {
			uchar hue = colorCode[col][0];
			uchar saturation = colorCode[col][1];
			uchar value = colorCode[col][2];
			float magnitude = normalizer * saturation * value; // magnitude is between 0 and 255
			BinData binData = color2bin[hue];
			binCodes[col][0] = binData.bin1;
			binCodes[col][2] = binData.bin2;
			binCodes[col][1] = cv::saturate_cast<uchar>(cvRound(binData.weight1 * magnitude));
			binCodes[col][3] = cv::saturate_cast<uchar>(cvRound(binData.weight2 * magnitude));
		}
	}
	return filtered;
}

} /* namespace imageprocessing */
