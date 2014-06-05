/*
 * GreyWorldNormalizationFilter.cpp
 *
 *  Created on: 30.07.2013
 *      Author: poschmann
 */

#include "imageprocessing/GreyWorldNormalizationFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Vec3b;
using cv::saturate_cast;
using std::invalid_argument;

namespace imageprocessing {

GreyWorldNormalizationFilter::GreyWorldNormalizationFilter() {}

Mat GreyWorldNormalizationFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.type() != CV_8UC3)
		throw invalid_argument("GreyWorldNormalizationFilter: The image type must be CV_8UC3");
	int rows = image.rows;
	int cols = image.cols;
	filtered.create(rows, cols, image.type());
	if (image.isContinuous() && filtered.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	double blueSum = 0, greenSum = 0, redSum = 0;
	uchar blueMax = 0, greenMax = 0, redMax = 0;
	for (int row = 0; row < rows; ++row) {
		const Vec3b* originalRow = image.ptr<Vec3b>(0);
		for (int col = 0; col < cols; ++col) {
			blueSum += originalRow[col][0];
			greenSum += originalRow[col][1];
			redSum += originalRow[col][2];
			if (originalRow[col][0] > blueMax)
				blueMax = originalRow[col][0];
			if (originalRow[col][1] > greenMax)
				greenMax = originalRow[col][1];
			if (originalRow[col][2] > redMax)
				redMax = originalRow[col][2];
		}
	}
	int n = rows * cols;
	double blueMean = static_cast<double>(blueSum) / n;
	double greenMean = static_cast<double>(greenSum) / n;
	double redMean = static_cast<double>(redSum) / n;
	double blueMaxNew = blueMax / blueMean;
	double greenMaxNew = greenMax / greenMean;
	double redMaxNew = redMax / redMean;
	double max = blueMaxNew;
	if (greenMaxNew > max)
		max = greenMaxNew;
	if (redMaxNew > max)
		max = redMaxNew;
	double blueScale = 255.0 / (blueMean * max);
	double greenScale = 255.0 / (greenMean * max);
	double redScale = 255.0 / (redMean * max);
	for (int row = 0; row < rows; ++row) {
		const Vec3b* originalRow = image.ptr<Vec3b>(0);
		Vec3b* filteredRow = filtered.ptr<Vec3b>(0);
		for (int col = 0; col < cols; ++col) {
			filteredRow[col][0] = saturate_cast<uchar>(cvRound(blueScale * originalRow[col][0]));
			filteredRow[col][1] = saturate_cast<uchar>(cvRound(greenScale * originalRow[col][1]));
			filteredRow[col][2] = saturate_cast<uchar>(cvRound(redScale * originalRow[col][2]));
		}
	}
	return filtered;
}

void GreyWorldNormalizationFilter::applyInPlace(Mat& image) const {
	applyTo(image, image);
}

} /* namespace imageprocessing */
