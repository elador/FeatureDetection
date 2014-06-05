/*
 * GradientSumFilter.cpp
 *
 *  Created on: 15.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/GradientSumFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Vec2b;
using std::invalid_argument;
using std::string;

namespace imageprocessing {

GradientSumFilter::GradientSumFilter(int rows, int cols) : rows(rows), cols(cols) {}

GradientSumFilter::GradientSumFilter(int count) : rows(count), cols(count) {}

Mat GradientSumFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.type() != CV_8UC2)
		throw invalid_argument("GradientSumFilter: the image must be of type CV_8UC2");
	if (image.rows % rows != 0)
		throw invalid_argument("GradientSumFilter: image row count (" + std::to_string(image.rows) + ") is not divisible by cell count (" + std::to_string(rows) + ")");
	if (image.cols % cols != 0)
		throw invalid_argument("GradientSumFilter: image column count (" + std::to_string(image.cols) + ") is not divisible by cell count (" + std::to_string(cols) + ")");

	filtered.create(1, rows * cols * 4, CV_32FC1);
	int cellWidth = image.cols / cols;
	int cellHeight = image.rows / rows;
	float* values = filtered.ptr<float>();
	float normalizer = 1.f / 127.f;
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			float sdx = 0;
			float sdy = 0;
			float sadx = 0;
			float sady = 0;
			for (int j = 0; j < cellHeight; ++j) {
				for (int i = 0; i < cellWidth; ++i) {
					const Vec2b& gradient = image.at<Vec2b>(row * cellHeight + j, col * cellWidth + i);
					float dx = normalizer * (static_cast<int>(gradient[0]) - 127);
					float dy = normalizer * (static_cast<int>(gradient[1]) - 127);
					sdx += dx;
					sdy += dy;
					sadx += fabs(dx);
					sady += fabs(dy);
				}
			}
			values[0] = sdx;
			values[1] = sdy;
			values[2] = sadx;
			values[3] = sady;
			values += 4;
		}
	}
	return filtered;
}

} /* namespace imageprocessing */
