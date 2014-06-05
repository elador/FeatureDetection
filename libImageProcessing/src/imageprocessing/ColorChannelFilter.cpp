/*
 * ColorChannelFilter.cpp
 *
 *  Created on: 09.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/ColorChannelFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Vec3b;
using std::invalid_argument;

namespace imageprocessing {

ColorChannelFilter::ColorChannelFilter(unsigned int bins, bool magnitude) : bins(bins), magnitude(magnitude) {
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

Mat ColorChannelFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.type() != CV_8UC3)
		throw invalid_argument("ColorChannelFilter: the image must be of type CV_8UC3");

	int rows = image.rows;
	int cols = image.cols;
	float normalizer = 1.f / 255.f;
	if (magnitude) { // include magnitude
		filtered.create(rows, cols, CV_8UC(bins + 1));
		if (image.isContinuous() && filtered.isContinuous()) {
			cols *= rows;
			rows = 1;
		}
		for (int row = 0; row < rows; ++row) {
			const Vec3b* colorCode = image.ptr<Vec3b>(row);
			uchar* values = filtered.data + filtered.step[0] * row;
			for (int col = 0; col < cols; ++col) {
				for (unsigned int bin = 0; bin <= bins; ++bin)
					values[bin] = 0;
				uchar hue = colorCode[col][0];
				uchar saturation = colorCode[col][1];
				uchar value = colorCode[col][2];
				float magnitude = normalizer * saturation * value; // magnitude is between 0 and 255
				BinData binData = color2bin[hue];
				values[binData.bin1] += cv::saturate_cast<uchar>(cvRound(binData.weight1 * magnitude));
				values[binData.bin2] += cv::saturate_cast<uchar>(cvRound(binData.weight2 * magnitude));
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
			const Vec3b* colorCode = image.ptr<Vec3b>(row);
			uchar* values = filtered.data + filtered.step[0] * row;
			for (int col = 0; col < cols; ++col) {
				for (unsigned int bin = 0; bin < bins; ++bin)
					values[bin] = 0;
				uchar hue = colorCode[col][0];
				uchar saturation = colorCode[col][1];
				uchar value = colorCode[col][2];
				float magnitude = normalizer * saturation * value; // magnitude is between 0 and 255
				BinData binData = color2bin[hue];
				values[binData.bin1] += cv::saturate_cast<uchar>(cvRound(binData.weight1 * magnitude));
				values[binData.bin2] += cv::saturate_cast<uchar>(cvRound(binData.weight2 * magnitude));
				values += bins;
			}
		}
	}

	return filtered;
}

} /* namespace imageprocessing */
