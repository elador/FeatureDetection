/*
 * FpdwFeaturesFilter.cpp
 *
 *  Created on: 09.02.2016
 *      Author: poschmann
 */

#include "imageprocessing/filtering/FpdwFeaturesFilter.hpp"
#include <stdexcept>
#include <vector>

using cv::Mat;
using cv::Vec2f;
using cv::Vec3b;
using cv::Vec3f;
using std::invalid_argument;

namespace imageprocessing {
namespace filtering {

FpdwFeaturesFilter::FpdwFeaturesFilter(bool fastGradient, bool interpolate, int normalizationRadius, double normalizationConstant) :
		grayConverter(),
		luvConverter(true),
		gradientFilter(1),
		magnitudeFilter(normalizationRadius, normalizationConstant),
		orientationFilter(false, normalizationRadius, normalizationConstant),
		fastGradient(fastGradient),
		binCount(6),
		value2bin(binCount / orientationFilter.getUpperBound()),
		interpolate(interpolate) {
	createGradientLut();
}

void FpdwFeaturesFilter::createGradientLut() {
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	// build the look-up table for gradient images of depth CV_8U
	// index of the look-up table is the binary concatenation of the gradients of x and y
	// values inside the look-up table are the bin indices and weights
	gradientCode.gradient.x = 0;
	for (int x = 0; x < 256; ++x) {
		float gradientX = (x - 127.f) / 255.f;
		gradientCode.gradient.y = 0;
		for (int y = 0; y < 256; ++y) {
			float gradientY = (y - 127.f) / 255.f;
			LutEntry entry;
			entry.magnitude = computeMagnitude(gradientX, gradientY);
			float orientation = computeOrientation(gradientX, gradientY);
			if (interpolate) {
				entry.fullBins = computeInterpolatedBins(orientation, 1);
			} else {
				entry.fullBins.bin1 = computeBin(orientation);
				entry.fullBins.weight1 = 1;
			}
			binLut[gradientCode.index] = entry;
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

Mat FpdwFeaturesFilter::applyTo(const Mat& bgrImage, Mat& descriptorImage) const {
	if (bgrImage.type() != CV_8UC3 && bgrImage.type() != CV_32FC3)
		throw invalid_argument("FpdwFeaturesFilter: the gradient image type must be CV_8UC3 or CV_32FC3, but was "
				+ std::to_string(bgrImage.type()));
	Mat gradientImage = computeGradientImage(bgrImage);
	Mat magnitudeImage = magnitudeFilter.applyTo(gradientImage);
	computeDescriptorImage(bgrImage, gradientImage, magnitudeImage, descriptorImage);
	return descriptorImage;
}

Mat FpdwFeaturesFilter::computeGradientImage(const Mat& bgrImage) const {
	if (fastGradient)
		return gradientFilter.applyTo(grayConverter.applyTo(bgrImage));
	else
		return reduceToStrongestGradient(gradientFilter.applyTo(bgrImage));
}

Mat FpdwFeaturesFilter::reduceToStrongestGradient(const Mat& gradientImage) const {
	Mat singleGradientImage(gradientImage.rows, gradientImage.cols, CV_MAKETYPE(gradientImage.depth(), 2));
	int originalChannels = gradientImage.channels() / 2;
	if (gradientImage.depth() == CV_8U) {
		for (int row = 0; row < gradientImage.rows; ++row) {
			for (int col = 0; col < gradientImage.cols; ++col) {
				const ushort* gradientCodes = gradientImage.ptr<ushort>(row, col); // concatenation of x gradient and y gradient (both uchar)
				ushort strongestGradientCode = gradientCodes[0];
				float strongestMagnitude = binLut[strongestGradientCode].magnitude;
				for (int ch = 1; ch < originalChannels; ++ch) {
					ushort gradientCode = gradientCodes[ch];
					if (strongestMagnitude < binLut[gradientCode].magnitude) {
						strongestGradientCode = gradientCode;
						strongestMagnitude = binLut[gradientCode].magnitude;
					}
				}
				singleGradientImage.at<ushort>(row, col) = strongestGradientCode;
			}
		}
	} else if (gradientImage.depth() == CV_32F) {
		for (int row = 0; row < gradientImage.rows; ++row) {
			for (int col = 0; col < gradientImage.cols; ++col) {
				const Vec2f* gradients = gradientImage.ptr<Vec2f>(row, col); // gradient for x and y
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
				singleGradientImage.at<Vec2f>(row, col) = strongestGradient;
			}
		}
	}
	return singleGradientImage;
}

void FpdwFeaturesFilter::computeDescriptorImage(const Mat& bgrImage,
		const Mat& gradientImage, const Mat& magnitudeImage, Mat& descriptorImage) const {
	int magnitudeOffset = binCount;
	int luvOffset = magnitudeOffset + 1;
	descriptorImage.create(gradientImage.rows, gradientImage.cols, CV_32FC(binCount + 1 + 3));
	if (gradientImage.depth() == CV_8U) {
		for (int row = 0; row < gradientImage.rows; ++row) {
			for (int col = 0; col < gradientImage.cols; ++col) {
				ushort gradientCode = gradientImage.at<ushort>(row, col); // concatenation of x gradient and y gradient (both uchar)
				float magnitude = magnitudeImage.at<float>(row, col);
				float* descriptor = descriptorImage.ptr<float>(row, col);
				for (int ch = 0; ch < binCount; ++ch)
					descriptor[ch] = 0;
				const LutEntry& entry = binLut[gradientCode];
				if (interpolate) {
					descriptor[entry.fullBins.bin1] = entry.fullBins.weight1 * magnitude;
					descriptor[entry.fullBins.bin2] = entry.fullBins.weight2 * magnitude;
				} else {
					descriptor[entry.fullBins.bin1] = magnitude;
				}
				descriptor[magnitudeOffset] = magnitude;
				Vec3b bgr = bgrImage.at<Vec3b>(row, col);
				Vec3f* luv = reinterpret_cast<Vec3f*>(descriptor + luvOffset);
				luvConverter.convertToNormalizedLuv(bgr, *luv);
			}
		}
	} else if (gradientImage.depth() == CV_32F) {
		Mat luvImage = luvConverter.applyTo(bgrImage);
		for (int row = 0; row < gradientImage.rows; ++row) {
			for (int col = 0; col < gradientImage.cols; ++col) {
				const Vec2f& gradient = gradientImage.at<Vec2f>(row, col); // gradient for x and y
				float orientation = computeOrientation(gradient[0], gradient[1]);
				float magnitude = magnitudeImage.at<float>(row, col);
				float* descriptor = descriptorImage.ptr<float>(row, col);
				for (int ch = 0; ch < binCount; ++ch)
					descriptor[ch] = 0;
				if (interpolate) {
					Bins bins = computeInterpolatedBins(orientation, magnitude);
					descriptor[bins.bin1] = bins.weight1;
					descriptor[bins.bin2] = bins.weight2;
				} else {
					int bin = computeBin(orientation);
					descriptor[bin] = magnitude;
				}
				descriptor[magnitudeOffset] = magnitude;
				Vec3f* luv = reinterpret_cast<Vec3f*>(descriptor + luvOffset);
				*luv = luvImage.at<Vec3f>(row, col);
			}
		}
	}
}

float FpdwFeaturesFilter::computeOrientation(float gradientX, float gradientY) const {
	return orientationFilter.computeOrientation(gradientX, gradientY);
}

float FpdwFeaturesFilter::computeMagnitude(float gradientX, float gradientY) const {
	return magnitudeFilter.computeMagnitude(gradientX, gradientY);
}

float FpdwFeaturesFilter::computeSquaredMagnitude(float gradientX, float gradientY) const {
	return magnitudeFilter.computeSquaredMagnitude(gradientX, gradientY);
}

int FpdwFeaturesFilter::computeBin(float value) const {
	int bin = static_cast<int>(value * value2bin + 0.5f);
	if (bin == binCount)
		bin = 0;
	return bin;
}

// bin1 and bin2 are guaranteed to be different
FpdwFeaturesFilter::Bins FpdwFeaturesFilter::computeInterpolatedBins(float value, float weight) const {
	Bins bins;
	const float bin = value * value2bin;
	bins.bin1 = static_cast<int>(bin);
	bins.bin2 = bins.bin1 + 1;
	if (bins.bin2 == binCount)
		bins.bin2 = 0;
	bins.weight2 = weight * (bin - bins.bin1);
	bins.weight1 = weight - bins.weight2;
	return bins;
}

} /* namespace filtering */
} /* namespace imageprocessing */
