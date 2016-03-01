/*
 * GradientOrientationFilter.cpp
 *
 *  Created on: 09.10.2015
 *      Author: poschmann
 */

#include "imageprocessing/filtering/GradientOrientationFilter.hpp"
#include <stdexcept>
#include <vector>

using cv::Mat;
using cv::Vec2f;
using std::invalid_argument;
using std::to_string;
using std::vector;

namespace imageprocessing {
namespace filtering {

GradientOrientationFilter::GradientOrientationFilter(bool full, int normalizationRadius, double normalizationConstant) :
		half(!full), magnitudeFilter(normalizationRadius, normalizationConstant) {
	createGradientLut();
}

void GradientOrientationFilter::createGradientLut() {
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	// build the look-up table for gradient images of depth CV_8U
	// index of the look-up table is the binary concatenation of the gradients of x and y
	// values inside the look-up table are the orientation and magnitude
	gradientCode.gradient.x = 0;
	for (int x = 0; x < 256; ++x) {
		float gradientX = (x - 127.f) / 255.f;
		gradientCode.gradient.y = 0;
		for (int y = 0; y < 256; ++y) {
			float gradientY = (y - 127.f) / 255.f;
			gradientLut[gradientCode.index][0] = computeOrientation(gradientX, gradientY);
			gradientLut[gradientCode.index][1] = computeMagnitude(gradientX, gradientY);
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

Mat GradientOrientationFilter::applyTo(const Mat& image, Mat& filtered) const {
	computeOrientationAndMagnitude(image, filtered);
	if (isNormalizing())
		normalizeMagnitude(filtered);
	return filtered;
}

void GradientOrientationFilter::computeOrientationAndMagnitude(const Mat& image, Mat& filtered) const {
	int channels = image.channels();
	if (channels < 2 || channels % 2 != 0)
		throw invalid_argument("GradientOrientationFilter: number of image channels must be even, but was " + to_string(channels));
	int originalChannels = channels / 2; // number of original channels (there are two gradients per original channel)
	filtered.create(image.rows, image.cols, CV_32FC2);
	if (image.depth() == CV_8U) {
		if (originalChannels == 1)
			computeOrientationAndMagnitudeForUchar(image, filtered);
		else
			computeOrientationAndMagnitudeForUcharN(image, originalChannels, filtered);
	} else if (image.depth() == CV_32F) {
		if (originalChannels == 1)
			computeOrientationAndMagnitudeForFloat(image, filtered);
		else
			computeOrientationAndMagnitudeForFloatN(image, originalChannels, filtered);
	} else {
		throw invalid_argument("GradientOrientationFilter: image depth must be CV_8U or CV_32F, but was " + to_string(image.depth()));
	}
}

void GradientOrientationFilter::computeOrientationAndMagnitudeForUchar(const Mat& image, Mat& filtered) const {
	for (int row = 0; row < image.rows; ++row) {
		for (int col = 0; col < image.cols; ++col) {
			ushort gradientCode = image.at<ushort>(row, col); // concatenation of x gradient and y gradient (both uchar)
			filtered.at<Vec2f>(row, col) = gradientLut[gradientCode]; // orientation and magnitude
		}
	}
}

void GradientOrientationFilter::computeOrientationAndMagnitudeForUcharN(const Mat& image, int originalChannels, Mat& filtered) const {
	for (int row = 0; row < image.rows; ++row) {
		for (int col = 0; col < image.cols; ++col) {
			const ushort* gradientCodes = image.ptr<ushort>(row, col); // concatenation of x gradient and y gradient (both uchar)
			Vec2f strongestGradient = gradientLut[gradientCodes[0]];
			for (int ch = 1; ch < originalChannels; ++ch) {
				ushort gradientCode = gradientCodes[ch]; // concatenation of x gradient and y gradient (both uchar)
				Vec2f gradient = gradientLut[gradientCode];
				if (gradient[1] > strongestGradient[1])
					strongestGradient = gradient;
			}
			filtered.at<Vec2f>(row, col) = strongestGradient; // orientation and magnitude
		}
	}
}

void GradientOrientationFilter::computeOrientationAndMagnitudeForFloat(const Mat& image, Mat& filtered) const {
	for (int row = 0; row < image.rows; ++row) {
		const Vec2f* gradients = image.ptr<Vec2f>(row); // gradient for x and y
		Vec2f* gradientOrientations = filtered.ptr<Vec2f>(row); // orientation and magnitude
		for (int col = 0; col < image.cols; ++col) {
			const Vec2f& gradient = gradients[col];
			gradientOrientations[col][0] = computeOrientation(gradient[0], gradient[1]);
			gradientOrientations[col][1] = computeMagnitude(gradient[0], gradient[1]);
		}
	}
}

void GradientOrientationFilter::computeOrientationAndMagnitudeForFloatN(const Mat& image, int originalChannels, Mat& filtered) const {
	for (int row = 0; row < image.rows; ++row) {
		const Vec2f* gradients = image.ptr<Vec2f>(row); // gradient for x and y
		Vec2f* gradientOrientations = filtered.ptr<Vec2f>(row); // orientation and magnitude
		for (int col = 0; col < image.cols; ++col) {
			Vec2f strongestGradient = gradients[0];
			float strongestSquaredMagnitude = computeSquaredMagnitude(strongestGradient[0], strongestGradient[1]);
			for (int ch = 1; ch < originalChannels; ++ch) {
				const Vec2f& gradient = gradients[ch];
				float squaredMagnitude = computeSquaredMagnitude(gradient[0], gradient[1]);
				if (squaredMagnitude > strongestSquaredMagnitude) {
					strongestSquaredMagnitude = squaredMagnitude;
					strongestGradient = gradient;
				}
			}
			gradientOrientations[col][0] = computeOrientation(strongestGradient[0], strongestGradient[1]);
			gradientOrientations[col][1] = std::sqrt(strongestSquaredMagnitude);
			gradients += originalChannels;
		}
	}
}

float GradientOrientationFilter::computeOrientation(float gradientX, float gradientY) const {
	float orientation = std::atan2(gradientY, gradientX); // orientation in [-pi,pi]
	if (orientation < 0) // orientation in [0,2*pi)
		orientation += TWO_PI;
	if (half && orientation >= PI) // orientation in [0,pi) for half
		orientation -= PI;
	return orientation;
}

float GradientOrientationFilter::computeMagnitude(float gradientX, float gradientY) const {
	return magnitudeFilter.computeMagnitude(gradientX, gradientY);
}

float GradientOrientationFilter::computeSquaredMagnitude(float gradientX, float gradientY) const {
	return magnitudeFilter.computeSquaredMagnitude(gradientX, gradientY);
}

bool GradientOrientationFilter::isNormalizing() const {
	return magnitudeFilter.isNormalizing();
}

void GradientOrientationFilter::normalizeMagnitude(Mat& orientationAndMagnitude) const {
	Mat magnitude = extractMagnitude(orientationAndMagnitude);
	Mat normalizer = magnitudeFilter.computeNormalizer(magnitude);
	normalizeMagnitude(orientationAndMagnitude, normalizer);
}

Mat GradientOrientationFilter::extractMagnitude(const Mat& orientationAndMagnitude) const {
	Mat magnitude(orientationAndMagnitude.rows, orientationAndMagnitude.cols, CV_32FC1);
	cv::mixChannels({orientationAndMagnitude}, {magnitude}, {1, 0});
	return magnitude;
}

void GradientOrientationFilter::normalizeMagnitude(Mat& orientationAndMagnitude, const Mat& normalizer) const {
	for (int row = 0; row < orientationAndMagnitude.rows; ++row) {
		for (int col = 0; col < orientationAndMagnitude.cols; ++col) {
			Vec2f& value = orientationAndMagnitude.at<Vec2f>(row, col);
			value[1] /= normalizer.at<float>(row, col);
		}
	}
}

} /* namespace filtering */
} /* namespace imageprocessing */
