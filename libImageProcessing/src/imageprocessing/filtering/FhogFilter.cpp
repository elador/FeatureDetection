/*
 * FhogFilter.cpp
 *
 *  Created on: 24.02.2016
 *      Author: poschmann
 */

#include "imageprocessing/filtering/FhogFilter.hpp"
#include <stdexcept>
#include <vector>

using cv::Mat;
using cv::Vec2f;
using std::invalid_argument;
using std::vector;

namespace imageprocessing {
namespace filtering {

FhogFilter::FhogFilter(int cellSize, int unsignedBinCount, bool interpolateBins, bool interpolateCells, float alpha) :
		magnitudeFilter(0),
		orientationFilter(true, 0),
		fhogAggregationFilter(cellSize, interpolateCells, alpha),
		cellSize(cellSize),
		signedBinCount(2 * unsignedBinCount),
		unsignedBinCount(unsignedBinCount),
		value2bin(signedBinCount / orientationFilter.getUpperBound()),
		interpolateBins(interpolateBins),
		interpolateCells(interpolateCells) {
	if (unsignedBinCount < 1)
		throw invalid_argument("FhogFilter: unsignedBinCount must be bigger than zero, but was: " + std::to_string(unsignedBinCount));
	createGradientLut();
}

void FhogFilter::createGradientLut() {
	// build the look-up table for gradients in [-255, 255], shifted to [1, 511]
	// index of the look-up table is the binary concatenation of the gradients of x and y
	// values inside the look-up table are the bin indices and weights
	for (int gradientCodeX = 1; gradientCodeX < 512; ++gradientCodeX) {
		float gradientX = (gradientCodeX - 256) / (255.0f * 2.0f);
		for (int gradientCodeY = 1; gradientCodeY < 512; ++gradientCodeY) {
			float gradientY = (gradientCodeY - 256) / (255.0f * 2.0f);
			LutEntry binLutEntry;
			binLutEntry.magnitude = computeMagnitude(gradientX, gradientY);
			float orientation = computeOrientation(gradientX, gradientY);
			if (interpolateBins) {
				binLutEntry.bins = computeInterpolatedBins(orientation, binLutEntry.magnitude);
			} else {
				binLutEntry.bins.index1 = computeBin(orientation);
				binLutEntry.bins.weight1 = binLutEntry.magnitude;
			}
			int gradientCode = gradientCodeY * 512 + gradientCodeX;
			binLut[gradientCode] = binLutEntry;
		}
	}
}

Mat FhogFilter::applyTo(const Mat& image, Mat& descriptors) const {
	if (image.type() != CV_8UC1 && image.type() != CV_8UC3)
		throw invalid_argument("FhogFilter: the image type must be CV_8UC1 or CV_8UC3, but was " + std::to_string(image.type()));
	int rows = image.rows / cellSize;
	int cols = image.cols / cellSize;
	int descriptorSize = signedBinCount + unsignedBinCount + 4;
	descriptors = Mat::zeros(rows, cols, CV_32FC(descriptorSize));
	if (image.channels() == 1)
		computeSignedHistograms<true>(image, descriptors);
	else
		computeSignedHistograms<false>(image, descriptors);
	fhogAggregationFilter.computeDescriptors(descriptors, descriptors, signedBinCount);
	return descriptors;
}

vector<FhogFilter::Coefficients> FhogFilter::computeInterpolationCoefficents(int sizeInPixels, int sizeInCells) const {
	vector<Coefficients> coefficients(sizeInPixels);
	if (interpolateCells) {
		for (int pixel = 0; pixel < sizeInPixels; ++pixel) {
			float realCellIndex = (pixel + 0.5f) / cellSize - 0.5f;
			int index1 = static_cast<int>(std::floor(realCellIndex));
			int index2 = index1 + 1;
			float weight2 = realCellIndex - index1;
			float weight1 = index2 - realCellIndex;
			if (index1 < 0) {
				index1 = index2;
				weight1 = 0;
			} else if (index2 >= sizeInCells) {
				index2 = index1;
				weight2 = 0;
			}
			coefficients[pixel] = { index1, index2, weight1, weight2 };
		}
	} else {
		for (int pixel = 0; pixel < sizeInPixels; ++pixel) {
			coefficients[pixel] = { pixel / cellSize, -1, 1, 0 };
		}
	}
	return coefficients;
}

float FhogFilter::computeOrientation(float gradientX, float gradientY) const {
	return orientationFilter.computeOrientation(gradientX, gradientY);
}

float FhogFilter::computeMagnitude(float gradientX, float gradientY) const {
	return magnitudeFilter.computeMagnitude(gradientX, gradientY);
}

int FhogFilter::computeBin(float value) const {
	int bin = static_cast<int>(value * value2bin + 0.5f);
	if (bin == signedBinCount)
		bin = 0;
	return bin;
}

// bin1 and bin2 are guaranteed to be different
FhogFilter::Coefficients FhogFilter::computeInterpolatedBins(float value, float weight) const {
	Coefficients bins;
	const float bin = value * value2bin;
	bins.index1 = static_cast<int>(bin);
	bins.index2 = bins.index1 + 1;
	if (bins.index2 == signedBinCount)
		bins.index2 = 0;
	bins.weight2 = weight * (bin - bins.index1);
	bins.weight1 = weight - bins.weight2;
	return bins;
}

cv::Mat FhogFilter::visualizeUnsignedHistograms(const cv::Mat& descriptors, int cellSize) {
	return FhogAggregationFilter::visualizeUnsignedHistograms(descriptors, cellSize);
}

} /* namespace filtering */
} /* namespace imageprocessing */
