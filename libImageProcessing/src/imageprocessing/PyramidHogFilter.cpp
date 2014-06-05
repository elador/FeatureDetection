/*
 * PyramidHogFilter.cpp
 *
 *  Created on: 05.09.2013
 *      Author: poschmann
 */

#include "imageprocessing/PyramidHogFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;

namespace imageprocessing {

PyramidHogFilter::PyramidHogFilter(int binCount, int levelCount, bool interpolate, bool signedAndUnsigned) :
		HistogramFilter(Normalization::L2NORM),
		binCount(binCount),
		maxLevel(levelCount - 1),
		histogramCount(0),
		interpolate(interpolate),
		signedAndUnsigned(signedAndUnsigned) {
	if (binCount <= 0)
		throw invalid_argument("PyramidHogFilter: binCount must be greater than zero");
	if (levelCount <= 0)
		throw invalid_argument("PyramidHogFilter: levelCount must be greater than zero");
	if (signedAndUnsigned && binCount % 2 != 0)
		throw invalid_argument("PyramidHogFilter: the bin size must be even for signed and unsigned gradients to be combined");
	for (int level = 0; level < levelCount; ++level)
		histogramCount += 1 << (2 * level);
}

Mat PyramidHogFilter::applyTo(const Mat& image, Mat& filtered) const {
	int realBinCount = signedAndUnsigned ? 3 * binCount / 2 : binCount;
	filtered = Mat::zeros(1, histogramCount, CV_32FC(realBinCount));
	float* cellHistogramsValues = filtered.ptr<float>() + (histogramCount - (1 << (2 * maxLevel))) * realBinCount;
	createCellHistograms(image, cellHistogramsValues, maxLevel, binCount, interpolate, signedAndUnsigned);
	for (int level = maxLevel - 1; level >= 0; --level) {
		float* blockHistogramsValues = cellHistogramsValues - (1 << (2 * level)) * realBinCount;
		combineHistograms(cellHistogramsValues, blockHistogramsValues, level, realBinCount);
		cellHistogramsValues = blockHistogramsValues;
	}
	normalizeHistograms(filtered.ptr<float>(), histogramCount, binCount, signedAndUnsigned);
	return filtered;
}

void PyramidHogFilter::createCellHistograms(
		const Mat& image, float* histogramsValues, int level, int binCount, bool interpolate, bool signedAndUnsigned) const {
	int gridCount = 1 << level;
	Mat cellHistograms;
	createCellHistograms(image, cellHistograms, binCount, gridCount, gridCount, interpolate);
	copyCellHistograms(cellHistograms.ptr<float>(), histogramsValues, binCount, gridCount * gridCount, signedAndUnsigned);
}

void PyramidHogFilter::copyCellHistograms(
		const float* source, float* destination, int binCount, int cellCount, bool signedAndUnsigned) const {
	const float* sourceHistogramValues = source;
	float* destinationHistogramValues = destination;
	int binHalfCount = binCount / 2;
	for (int histogramIndex = 0; histogramIndex < cellCount; ++histogramIndex) {
		for (int binIndex = 0; binIndex < binCount; ++binIndex)
			destinationHistogramValues[binIndex] = sourceHistogramValues[binIndex];
		destinationHistogramValues += binCount;
		if (signedAndUnsigned) {
			for (int binIndex = 0; binIndex < binHalfCount; ++binIndex)
				destinationHistogramValues[binIndex] = sourceHistogramValues[binIndex] + sourceHistogramValues[binHalfCount + binIndex];
			destinationHistogramValues += binHalfCount;
		}
		sourceHistogramValues += binCount;
	}
}

void PyramidHogFilter::combineHistograms(
		const float* cellHistogramsValues, float* blockHistogramsValues, int level, int binCount) const {
	int blockCount = 1 << level;
	int cellCount = blockCount << 1;
	float* blockHistogramValues = blockHistogramsValues;
	for (int blockRow = 0; blockRow < blockCount; ++blockRow) {
		for (int blockCol = 0; blockCol < blockCount; ++blockCol) {
			Mat blockHistogram(1, binCount, CV_32F, blockHistogramValues);
			for (int cellRow = 2 * blockRow; cellRow < 2 * (blockRow + 1); ++cellRow) {
				for (int cellCol = 2 * blockCol; cellCol < 2 * (blockCol + 1); ++cellCol) {
					unsigned int cellIndex = cellRow * cellCount + cellCol;
					const float* cellHistogramValues = cellHistogramsValues + cellIndex * binCount;
					for (int bin = 0; bin < binCount; ++bin)
						blockHistogramValues[bin] += cellHistogramValues[bin];
				}
			}
			blockHistogramValues += binCount;
		}
	}
}

void PyramidHogFilter::normalizeHistograms(float* histogramsValues, int histogramCount, int binCount, bool signedAndUnsigned) const {
	int binHalfCount = binCount / 2;
	int realBinCount = signedAndUnsigned ? binCount + binHalfCount : binCount;
	float* histogramValues = histogramsValues;
	for (int histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex) {
		// compute gradient energy
		float energy = 0;
		if (signedAndUnsigned) {
			for (int binIndex = binCount; binIndex < realBinCount; ++binIndex)
				energy += histogramValues[binIndex] * histogramValues[binIndex];
		} else {
			for (int binIndex = 0; binIndex < binCount; ++binIndex)
				energy += histogramValues[binIndex] * histogramValues[binIndex];
		}
		// normalize histogram
		float normalizer = 1.f / sqrt(energy + eps);
		for (int binIndex = 0; binIndex < realBinCount; ++binIndex)
			histogramValues[binIndex] = normalizer * histogramValues[binIndex];
		histogramValues += realBinCount;
	}
}

} /* namespace imageprocessing */
