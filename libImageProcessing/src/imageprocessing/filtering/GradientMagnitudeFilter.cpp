/*
 * GradientMagnitudeFilter.cpp
 *
 *  Created on: 07.01.2016
 *      Author: poschmann
 */

#include "imageprocessing/filtering/GradientMagnitudeFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Point;
using cv::Vec2f;
using std::invalid_argument;
using std::to_string;

namespace imageprocessing {
namespace filtering {

GradientMagnitudeFilter::GradientMagnitudeFilter(int normalizationRadius, double normalizationConstant) :
		smoothingFilter(2 * normalizationRadius + 1, 1, 1, normalizationConstant) {
	if (normalizationConstant <= 0)
		throw invalid_argument("GradientMagnitudeFilter: normalizationConstant must be bigger than zero, but was "
				+ std::to_string(normalizationConstant));
	createMagnitudeLut();
}

void GradientMagnitudeFilter::createMagnitudeLut() {
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	// build the look-up table for gradient images of depth CV_8U
	// index of the look-up table is the binary concatenation of the gradients of x and y
	// value inside the look-up table is magnitude
	gradientCode.gradient.x = 0;
	for (int x = 0; x < 256; ++x) {
		float gradientX = (x - 127.f) / 255.f;
		gradientCode.gradient.y = 0;
		for (int y = 0; y < 256; ++y) {
			float gradientY = (y - 127.f) / 255.f;
			magnitudeLut[gradientCode.index] = computeMagnitude(gradientX, gradientY);
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

Mat GradientMagnitudeFilter::applyTo(const Mat& image, Mat& filtered) const {
	computeMagnitudeImage(image, filtered);
	if (isNormalizing())
		normalizeMagnitude(filtered);
	return filtered;
}

void GradientMagnitudeFilter::computeMagnitudeImage(const Mat& gradients, Mat& magnitude) const {
	int channels = gradients.channels();
	if (channels < 2 || channels % 2 != 0)
		throw invalid_argument("GradientMagnitudeFilter: number of image channels must be even, but was " + to_string(channels));
	int originalChannels = channels / 2; // number of original channels (there are two gradients per original channel)
	magnitude.create(gradients.rows, gradients.cols, CV_32FC(originalChannels));
	if (gradients.depth() == CV_8U) {
		if (originalChannels == 1)
			computeMagnitudeImage_<ushort>(gradients, magnitude);
		else
			computeMagnitudesImage_<ushort>(gradients, magnitude);
	} else if (gradients.depth() == CV_32F) {
		if (originalChannels == 1)
			computeMagnitudeImage_<Vec2f>(gradients, magnitude);
		else
			computeMagnitudesImage_<Vec2f>(gradients, magnitude);
	} else {
		throw invalid_argument("GradientMagnitudeFilter: image depth must be CV_8U or CV_32F, but was " + to_string(gradients.depth()));
	}
}

float GradientMagnitudeFilter::computeMagnitude(float gradientX, float gradientY) const {
	return std::sqrt(computeSquaredMagnitude(gradientX, gradientY));
}

float GradientMagnitudeFilter::computeSquaredMagnitude(float gradientX, float gradientY) const {
	return gradientX * gradientX + gradientY * gradientY;
}

bool GradientMagnitudeFilter::isNormalizing() const {
	return smoothingFilter.getSize() > 1;
}

void GradientMagnitudeFilter::normalizeMagnitude(Mat& magnitude) const {
	Mat normalizer = computeNormalizer(magnitude);
	normalizeMagnitude(magnitude, normalizer);
}

Mat GradientMagnitudeFilter::computeNormalizer(const Mat& magnitude) const {
	return smoothingFilter.applyTo(magnitude);
}

void GradientMagnitudeFilter::normalizeMagnitude(Mat& magnitude, const Mat& normalizer) const {
	if (magnitude.channels() == 1)
		normalizeMagnitude1(magnitude, normalizer);
	else
		normalizeMagnitudeN(magnitude, normalizer);
}

void GradientMagnitudeFilter::normalizeMagnitude1(Mat& magnitude, const Mat& normalizer) const {
	for (int row = 0; row < magnitude.rows; ++row) {
		for (int col = 0; col < magnitude.cols; ++col) {
			magnitude.at<float>(row, col) /= normalizer.at<float>(row, col);
		}
	}
}

void GradientMagnitudeFilter::normalizeMagnitudeN(Mat& magnitude, const Mat& normalizer) const {
	int channels = magnitude.channels();
	for (int row = 0; row < magnitude.rows; ++row) {
		for (int col = 0; col < magnitude.cols; ++col) {
			const float* norm = normalizer.ptr<float>(row, col);
			float* values = magnitude.ptr<float>(row, col);
			for (int ch = 0; ch < channels; ++ch) {
				values[ch] /= norm[ch];
			}
		}
	}
}

} /* namespace filtering */
} /* namespace imageprocessing */
