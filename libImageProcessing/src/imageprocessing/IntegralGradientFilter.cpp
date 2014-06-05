/*
 * IntegralGradientFilter.cpp
 *
 *  Created on: 14.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/IntegralGradientFilter.hpp"
#include <stdexcept>
#include <vector>

using cv::Mat;
using cv::Vec2b;
using std::invalid_argument;
using std::vector;

namespace imageprocessing {

IntegralGradientFilter::IntegralGradientFilter(int rows, int cols) : rows(rows), cols(cols) {}

IntegralGradientFilter::IntegralGradientFilter(int count) : rows(count), cols(count) {}

Mat IntegralGradientFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.type() != CV_32SC1)
		throw invalid_argument("IntegralGradientFilter: the image must be of type CV_32SC1");

	filtered.create(rows, cols, CV_8UC2);
	int width = image.cols - 1;
	int height = image.rows - 1;

	int radiusX = std::max(1, cvRound(static_cast<double>(width - 1) / static_cast<double>(cols + 2)));
	int radiusY = std::max(1, cvRound(static_cast<double>(height - 1) / static_cast<double>(rows + 2)));
	double spacingX = static_cast<double>(width - 3 * radiusX) / static_cast<double>(cols - 1);
	double spacingY = static_cast<double>(height - 3 * radiusY) / static_cast<double>(rows - 1);
	int gradientHalfArea = radiusY * radiusX;

	int step0 = static_cast<int>(image.step.p[0]);
	int step1 = static_cast<int>(image.step.p[1]);
	int offsetY[] = { -radiusY * step0, 0, radiusY * step0, 2 * radiusY * step0 };
	int offsetX[] = { -radiusX * step1, 0, radiusX * step1, 2 * radiusX * step1 };
	vector<size_t> rowOffset(rows);
	vector<size_t> colOffset(cols);
	for (int row = 0; row < rows; ++row)
		rowOffset[row] = cvRound(radiusY + row * spacingY) * image.step.p[0];
	for (int col = 0; col < cols; ++col)
		colOffset[col] = cvRound(radiusX + col * spacingX) * image.step.p[1];

	for (int row = 0; row < rows; ++row) {
		const uchar* pRow = image.data + rowOffset[row];
		Vec2b* gradientValues = filtered.ptr<Vec2b>(row);
		for (int col = 0; col < cols; ++col) {
			const uchar* pPoint = pRow + colOffset[col];
			Vec2b& gradient = gradientValues[col];

			//     p1  p2
			// p3  p4  p5  p6
			// p7  p8  p9  p10
			//     p11 p12
			int p1  = *reinterpret_cast<const int*>(pPoint + offsetY[0] + offsetX[1]);
			int p2  = *reinterpret_cast<const int*>(pPoint + offsetY[0] + offsetX[2]);
			int p3  = *reinterpret_cast<const int*>(pPoint + offsetY[1] + offsetX[0]);
			int p4  = *reinterpret_cast<const int*>(pPoint + offsetY[1] + offsetX[1]);
			int p5  = *reinterpret_cast<const int*>(pPoint + offsetY[1] + offsetX[2]);
			int p6  = *reinterpret_cast<const int*>(pPoint + offsetY[1] + offsetX[3]);
			int p7  = *reinterpret_cast<const int*>(pPoint + offsetY[2] + offsetX[0]);
			int p8  = *reinterpret_cast<const int*>(pPoint + offsetY[2] + offsetX[1]);
			int p9  = *reinterpret_cast<const int*>(pPoint + offsetY[2] + offsetX[2]);
			int p10 = *reinterpret_cast<const int*>(pPoint + offsetY[2] + offsetX[3]);
			int p11 = *reinterpret_cast<const int*>(pPoint + offsetY[3] + offsetX[1]);
			int p12 = *reinterpret_cast<const int*>(pPoint + offsetY[3] + offsetX[2]);

			int top = p1 - p2 - p4 + p5;
			int bottom = p8 - p9 - p11 + p12;
			int left = p3 - p4 - p7 + p8;
			int right = p5 - p6 - p9 + p10;

			int dx = (right - left) / (2 * gradientHalfArea);
			int dy = (bottom - top) / (2 * gradientHalfArea);
			gradient[0] = static_cast<uchar>(dx + 127);
			gradient[1] = static_cast<uchar>(dy + 127);
		}
	}

	return filtered;
}

} /* namespace imageprocessing */
