/*
 * IntegralImageFilter.cpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/IntegralImageFilter.hpp"

namespace imageprocessing {

IntegralImageFilter::IntegralImageFilter(bool squared) : squared(squared) {}

IntegralImageFilter::~IntegralImageFilter() {}

Mat IntegralImageFilter::applyTo(const Mat& image, Mat& filtered) {
	// TODO replace by cv::integral (and change documentation of the class accordingly because of the trailing 0-row and -column)
	if (image.type() != CV_8U)
		throw "the input image must be of type CV_8U";
	filtered.create(image.rows, image.cols, CV_32F);

	float rowSum;
	if (squared) {
		const uchar* inRow = image.ptr<uchar>(0);
		float* outRow = filtered.ptr<float>(0);
		rowSum = 0;
		for (int c = 0; c < image.cols; ++c) {
			rowSum += inRow[c] * inRow[c];
			outRow[c] = rowSum;
		}
		for (int r = 1; r < image.rows; ++r) {
			float* prevOutRow = outRow;
			inRow = image.ptr<uchar>(r);
			outRow = filtered.ptr<float>(r);
			rowSum = 0;
			for (int c = 0; c < image.cols; ++c) {
				rowSum += inRow[c] * inRow[c];
				outRow[c] = prevOutRow[c] + rowSum;
			}
		}
	} else {
		const uchar* inRow = image.ptr<uchar>(0);
		float* outRow = filtered.ptr<float>(0);
		rowSum = 0;
		for (int c = 0; c < image.cols; ++c) {
			rowSum += inRow[c];
			outRow[c] = rowSum;
		}
		for (int r = 1; r < image.rows; ++r) {
			float* prevOutRow = outRow;
			inRow = image.ptr<uchar>(r);
			outRow = filtered.ptr<float>(r);
			rowSum = 0;
			for (int c = 0; c < image.cols; ++c) {
				rowSum += inRow[c];
				outRow[c] = prevOutRow[c] + rowSum;
			}
		}
	}
	return filtered;
}

void IntegralImageFilter::applyInPlace(Mat& image) {
	Mat filtered = applyTo(image);
	image = filtered;
}

} /* namespace imageprocessing */
