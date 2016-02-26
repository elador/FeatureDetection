/*
 * HistogramFilter.cpp
 *
 *  Created on: 13.10.2015
 *      Author: poschmann
 */

#include "imageprocessing/filtering/HistogramFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Vec2f;
using std::invalid_argument;

namespace imageprocessing {
namespace filtering {

HistogramFilter::HistogramFilter(int binCount, float upperBound, bool circular, bool interpolate) :
		binCount(binCount), value2bin(binCount / upperBound), circular(circular), interpolate(binCount > 1 ? interpolate : false) {
	if (binCount < 1)
		throw invalid_argument("HistogramFilter: binCount must be bigger than zero, but was " + std::to_string(binCount));
	if (upperBound <= 0)
		throw invalid_argument("HistogramFilter: upperBound must be bigger than zero, but was " + std::to_string(upperBound));
}

Mat HistogramFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.depth() != CV_32F)
		throw invalid_argument("HistogramFilter: image must have a depth of CV_32F, but had " + std::to_string(image.depth()));
	filtered.create(image.rows, image.cols, CV_32FC(binCount));
	if (image.channels() == 1) {
		for (int row = 0; row < image.rows; ++row) {
			for (int col = 0; col < image.cols; ++col) {
				float value = image.at<float>(row, col);
				float* histogram = filtered.ptr<float>(row, col);
				computeHistogram(histogram, value, 1.0f);
			}
		}
	} else if (image.channels() == 2) {
		for (int row = 0; row < image.rows; ++row) {
			for (int col = 0; col < image.cols; ++col) {
				Vec2f valueAndWeight = image.at<Vec2f>(row, col);
				float* histogram = filtered.ptr<float>(row, col);
				computeHistogram(histogram, valueAndWeight[0], valueAndWeight[1]);
			}
		}
	} else {
		throw invalid_argument("HistogramFilter: image must have one or two channels, but had " + std::to_string(image.channels()));
	}
	return filtered;
}

void HistogramFilter::computeHistogram(float* histogram, float value, float weight) const {
	for (int bin = 0; bin < binCount; ++bin)
		histogram[bin] = 0;
	if (interpolate) {
		Bins bins = computeInterpolatedBins(value, weight);
		histogram[bins.bin1] = bins.weight1;
		histogram[bins.bin2] = bins.weight2;
	} else {
		int bin = computeBin(value);
		histogram[bin] = weight;
	}
}

int HistogramFilter::computeBin(float value) const {
	if (circular) {
		return computeCircularBin(value);
	} else {
		return computeNoncircularBin(value);
	}
}

int HistogramFilter::computeCircularBin(float value) const {
	int bin = static_cast<int>(value * value2bin + 0.5f);
	if (bin == binCount)
		bin = 0;
	return bin;
}

int HistogramFilter::computeNoncircularBin(float value) const {
	return static_cast<int>(value * value2bin);
}

// bin1 and bin2 are guaranteed to be different
HistogramFilter::Bins HistogramFilter::computeInterpolatedBins(float value, float weight) const {
	if (circular)
		return computeCircularInterpolatedBins(value, weight);
	else
		return computeNoncircularInterpolatedBins(value, weight);
}

// bin1 and bin2 are guaranteed to be different
HistogramFilter::Bins HistogramFilter::computeCircularInterpolatedBins(float value, float weight) const {
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

// bin1 and bin2 are guaranteed to be different
HistogramFilter::Bins HistogramFilter::computeNoncircularInterpolatedBins(float value, float weight) const {
	Bins bins;
	const float bin = value * value2bin - 0.5f;
	bins.bin1 = static_cast<int>(bin);
	bins.bin2 = bins.bin1 + 1;
	if (bins.bin1 == -1) {
		bins.bin1 = binCount - 1;
		bins.weight1 = 0;
		bins.weight2 = weight;
	} else if (bins.bin2 == binCount) {
		bins.bin2 = 0;
		bins.weight2 = 0;
		bins.weight1 = weight;
	} else {
		bins.weight2 = weight * (bin - bins.bin1);
		bins.weight1 = weight - bins.weight2;
	}
	return bins;
}

} /* namespace filtering */
} /* namespace imageprocessing */
