/*
 * SpatialPyramidHistogramFilter.cpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/SpatialPyramidHistogramFilter.hpp"
#include <vector>
#include <stdexcept>

using cv::Mat;
using cv::Vec2b;
using cv::Vec4b;
using std::vector;
using std::runtime_error;
using std::invalid_argument;

namespace imageprocessing {

SpatialPyramidHistogramFilter::SpatialPyramidHistogramFilter(
		int binCount, int levelCount, bool interpolate, Normalization normalization) :
				HistogramFilter(normalization),
				binCount(binCount),
				maxLevel(levelCount - 1),
				histogramCount(0),
				interpolate(interpolate) {
	if (binCount <= 0)
		throw invalid_argument("SpatialPyramidHistogramFilter: binCount must be greater than zero");
	if (levelCount <= 0)
		throw invalid_argument("SpatialPyramidHistogramFilter: levelCount must be greater than zero");
	for (int level = 0; level < levelCount; ++level)
		histogramCount += 1 << (2 * level);
}

Mat SpatialPyramidHistogramFilter::applyTo(const Mat& image, Mat& filtered) const {
	filtered = Mat::zeros(1, histogramCount, CV_32FC(binCount));
	float* cellHistogramsValues = filtered.ptr<float>() + histogramCount * binCount - (1 << (2 * maxLevel)) * binCount;
	createCellHistograms(image, cellHistogramsValues, maxLevel, binCount, interpolate);
	for (int level = maxLevel - 1; level >= 0; --level) {
		float* blockHistogramsValues = cellHistogramsValues - (1 << (2 * level)) * binCount;
		combineHistograms(cellHistogramsValues, blockHistogramsValues, level, binCount);
		cellHistogramsValues = blockHistogramsValues;
	}
	normalizeHistograms(filtered.ptr<float>(), histogramCount, binCount);
	return filtered;
}

void SpatialPyramidHistogramFilter::createCellHistograms(
		const Mat& image, float* histogramsValues, int level, int binCount, bool interpolate) const {
	int histogramCount = 1 << level;
	Mat cellHistograms(histogramCount, histogramCount, CV_32FC(binCount), histogramsValues);
	createCellHistograms(image, cellHistograms, binCount, histogramCount, histogramCount, interpolate);
}

void SpatialPyramidHistogramFilter::combineHistograms(
		const float* cellHistogramsValues, float* blockHistogramsValues, int level, int binCount) const {
	int blockCount = 1 << level;
	int cellCount = blockCount << 1;
	float* blockHistogramValues = blockHistogramsValues;
	for (int blockRow = 0; blockRow < blockCount; ++blockRow) {
		for (int blockCol = 0; blockCol < blockCount; ++blockCol) {
			Mat blockHistogram(1, binCount, CV_32F, blockHistogramValues);
			for (int cellRow = 2 * blockRow; cellRow < 2 * (blockRow + 1); ++cellRow) {
				for (int cellCol = 2 * blockCol; cellCol < 2 * (blockCol + 1); ++cellCol) {
					const float* cellHistogramValues = cellHistogramsValues + (cellRow * cellCount + cellCol) * binCount;
					for (int bin = 0; bin < binCount; ++bin)
						blockHistogramValues[bin] += cellHistogramValues[bin];
				}
			}
			blockHistogramValues += binCount;
		}
	}
}

void SpatialPyramidHistogramFilter::normalizeHistograms(float* histogramsValues, int histogramCount, int binCount) const {
	for (int i = 0; i < histogramCount; ++i) {
		Mat histogram(1, binCount, CV_32F, histogramsValues + i * binCount);
		normalize(histogram);
	}
}

} /* namespace imageprocessing */
