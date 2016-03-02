/*
 * FhogAggregationFilter.cpp
 *
 *  Created on: 17.02.2016
 *      Author: poschmann
 */

#include "imageprocessing/filtering/FhogAggregationFilter.hpp"
#include "imageprocessing/filtering/GradientHistogramFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdexcept>

using cv::Mat;
using std::array;
using std::invalid_argument;
using std::vector;

namespace imageprocessing {
namespace filtering {

const float FhogAggregationFilter::eps = 1e-4;

FhogAggregationFilter::FhogAggregationFilter(int cellSize, bool interpolate, float alpha) :
		alpha(alpha), aggregationFilter(cellSize, interpolate, false) {
	if (alpha <= 0)
		throw invalid_argument("FhogAggregationFilter: alpha must be bigger than zero, but was: " + std::to_string(alpha));
}

Mat FhogAggregationFilter::applyTo(const Mat& image, Mat& descriptors) const {
	if (image.depth() != CV_32F)
		throw invalid_argument("FhogAggregationFilter: the image must have a depth of CV_32F, but had " + std::to_string(image.depth()));
	if (image.channels() % 2 != 0)
		throw invalid_argument("FhogAggregationFilter: the image must have an even number of channels, but had " + std::to_string(image.channels()));
	Mat histograms = aggregationFilter.applyTo(image);
	int signedBinCount = histograms.channels();
	computeDescriptors(descriptors, histograms, signedBinCount);
	return descriptors;
}

void FhogAggregationFilter::computeDescriptors(Mat& descriptors, const Mat& histograms, int signedBinCount) const {
	Mat energies = computeGradientEnergies(histograms, signedBinCount);
	computeDescriptors(descriptors, histograms, energies, signedBinCount);
}

Mat FhogAggregationFilter::computeGradientEnergies(const Mat& histograms, int signedBinCount) const {
	int unsignedBinCount = signedBinCount / 2;
	Mat energies(histograms.rows, histograms.cols, CV_32FC1);
	for (int row = 0; row < histograms.rows; ++row) {
		for (int col = 0; col < histograms.cols; ++col) {
			const float* signedHistogram = histograms.ptr<float>(row, col);
			energies.at<float>(row, col) = computeGradientEnergy(signedHistogram, unsignedBinCount);
		}
	}
	return energies;
}

float FhogAggregationFilter::computeGradientEnergy(const float* signedHistogram, int unsignedBinCount) const {
	float energy = 0;
	for (int bin = 0; bin < unsignedBinCount; ++bin) {
		int oppositeBin = bin + unsignedBinCount;
		float unsignedBinValue = signedHistogram[bin] + signedHistogram[oppositeBin];
		energy += unsignedBinValue * unsignedBinValue;
	}
	return energy;
}

void FhogAggregationFilter::computeDescriptors(Mat& descriptors, const Mat& histograms, const Mat& energies, int signedBinCount) const {
	int unsignedBinCount = signedBinCount / 2;
	int descriptorSize = signedBinCount + unsignedBinCount + 4;
	descriptors.create(histograms.rows, histograms.cols, CV_32FC(descriptorSize));
	for (int row = 0; row < histograms.rows; ++row) {
		for (int col = 0; col < histograms.cols; ++col) {
			array<float, 4> normalizers = computeNormalizers(energies, row, col);
			computeDescriptor(descriptors.ptr<float>(row, col),
					histograms.ptr<float>(row, col), normalizers, signedBinCount, unsignedBinCount);
		}
	}
}

array<float, 4> FhogAggregationFilter::computeNormalizers(const Mat& energies, int currRow, int currCol) const {
	int prevRow = std::max(currRow - 1, 0);
	int nextRow = std::min(currRow + 1, energies.rows - 1);
	int prevCol = std::max(currCol - 1, 0);
	int nextCol = std::min(currCol + 1, energies.cols - 1);
	float ulEnergySq = energies.at<float>(prevRow, prevCol); // up left
	float ucEnergySq = energies.at<float>(prevRow, currCol); // up center
	float urEnergySq = energies.at<float>(prevRow, nextCol); // up right
	float clEnergySq = energies.at<float>(currRow, prevCol); // center left
	float ccEnergySq = energies.at<float>(currRow, currCol); // center center
	float crEnergySq = energies.at<float>(currRow, nextCol); // center right
	float dlEnergySq = energies.at<float>(nextRow, prevCol); // down left
	float dcEnergySq = energies.at<float>(nextRow, currCol); // down center
	float drEnergySq = energies.at<float>(nextRow, nextCol); // down right
	return {
			1.f / std::sqrt(ulEnergySq + ucEnergySq + clEnergySq + ccEnergySq + eps),
			1.f / std::sqrt(ucEnergySq + urEnergySq + ccEnergySq + crEnergySq + eps),
			1.f / std::sqrt(clEnergySq + ccEnergySq + dlEnergySq + dcEnergySq + eps),
			1.f / std::sqrt(ccEnergySq + crEnergySq + dcEnergySq + drEnergySq + eps)
	};
}

void FhogAggregationFilter::computeDescriptor(float* descriptor, const float* signedHistogram,
		const array<float, 4>& normalizers, int signedBinCount, int unsignedBinCount) const {
	// "magic numbers" and general structure taken from [2] (see header file) with the following license:
	//
	// Copyright (C) 2011, 2012 Ross Girshick, Pedro Felzenszwalb
	// Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
	// Copyright (C) 2007 Pedro Felzenszwalb, Deva Ramanan
	//
	// Permission is hereby granted, free of charge, to any person obtaining
	// a copy of this software and associated documentation files (the
	// "Software"), to deal in the Software without restriction, including
	// without limitation the rights to use, copy, modify, merge, publish,
	// distribute, sublicense, and/or sell copies of the Software, and to
	// permit persons to whom the Software is furnished to do so, subject to
	// the following conditions:
	//
	// The above copyright notice and this permission notice shall be
	// included in all copies or substantial portions of the Software.
	//
	// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
	// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
	// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
	// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
	// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
	// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
	// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	array<float, 4> energy = { 0, 0, 0, 0 };
	// unsigned orientation features (aka contrast-insensitive)
	for (int bin = 0; bin < unsignedBinCount; ++bin) {
		int oppositeBin = bin + unsignedBinCount;
		int unsignedBin = signedBinCount + bin;
		float unsignedBinValue = signedHistogram[bin] + signedHistogram[oppositeBin];
		array<float, 4> normalizedValues = computeNormalizedValues(unsignedBinValue, normalizers);
		descriptor[unsignedBin] = 0.5 * computeSum(normalizedValues);
	}
	// signed orientation features (aka contrast-sensitive)
	for (int bin = 0; bin < signedBinCount; ++bin) {
		array<float, 4> normalizedValues = computeNormalizedValues(signedHistogram[bin], normalizers);
		descriptor[bin] = 0.5 * computeSum(normalizedValues);
		addTo(energy, normalizedValues);
	}
	// energy features (aka texture features)
	descriptor[signedBinCount + unsignedBinCount]     = 0.2357 * energy[0];
	descriptor[signedBinCount + unsignedBinCount + 1] = 0.2357 * energy[1];
	descriptor[signedBinCount + unsignedBinCount + 2] = 0.2357 * energy[2];
	descriptor[signedBinCount + unsignedBinCount + 3] = 0.2357 * energy[3];
}

array<float, 4> FhogAggregationFilter::computeNormalizedValues(float value, const array<float, 4>& normalizers) const {
	return {
			std::min(alpha, normalizers[0] * value),
			std::min(alpha, normalizers[1] * value),
			std::min(alpha, normalizers[2] * value),
			std::min(alpha, normalizers[3] * value)
	};
}

float FhogAggregationFilter::computeSum(const array<float, 4>& values) const {
	return values[0] + values[1] + values[2] + values[3];
}

void FhogAggregationFilter::addTo(array<float, 4>& sums, const array<float, 4>& values) const {
	sums[0] += values[0];
	sums[1] += values[1];
	sums[2] += values[2];
	sums[3] += values[3];
}

Mat FhogAggregationFilter::visualizeUnsignedHistograms(const Mat& descriptors, int cellSize) {
	if (descriptors.channels() < 7)
		throw invalid_argument("FhogAggregationFilter: descriptors image must have at least 7 channels, but had "
				+ std::to_string(descriptors.channels()));
	int unsignedBinCount = (descriptors.channels() - 4) / 3;
	return GradientHistogramFilter::visualizeUnsignedHistograms(descriptors, unsignedBinCount, 2 * unsignedBinCount, cellSize);
}

} /* namespace filtering */
} /* namespace imageprocessing */
