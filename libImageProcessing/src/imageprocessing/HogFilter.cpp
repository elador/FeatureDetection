/*
 * HogFilter.cpp
 *
 *  Created on: 26.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/HogFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;

namespace imageprocessing {

HogFilter::HogFilter(int binCount, int cellSize, int blockSize, bool interpolate, bool signedAndUnsigned) :
				HistogramFilter(Normalization::L2NORM),
				binCount(binCount),
				cellWidth(cellSize),
				cellHeight(cellSize),
				blockWidth(blockSize),
				blockHeight(blockSize),
				interpolate(interpolate),
				signedAndUnsigned(signedAndUnsigned) {
	if (binCount <= 0)
		throw invalid_argument("HogFilter: binCount must be greater than zero");
	if (cellSize <= 0)
		throw invalid_argument("HogFilter: cellSize must be greater than zero");
	if (blockSize <= 0)
		throw invalid_argument("HogFilter: blockSize must be greater than zero");
	if (signedAndUnsigned && binCount % 2 != 0)
		throw invalid_argument("HogFilter: the bin size must be even for signed and unsigned gradients to be combined");
}

HogFilter::HogFilter(int binCount, int cellWidth, int cellHeight, int blockWidth, int blockHeight, bool interpolate, bool signedAndUnsigned) :
				HistogramFilter(Normalization::L2NORM),
				binCount(binCount),
				cellWidth(cellWidth),
				cellHeight(cellHeight),
				blockWidth(blockWidth),
				blockHeight(blockHeight),
				interpolate(interpolate),
				signedAndUnsigned(signedAndUnsigned) {
	if (binCount <= 0)
		throw invalid_argument("HogFilter: binCount must be greater than zero");
	if (cellWidth <= 0)
		throw invalid_argument("HogFilter: cellWidth must be greater than zero");
	if (cellHeight <= 0)
		throw invalid_argument("HogFilter: cellHeight must be greater than zero");
	if (blockWidth <= 0)
		throw invalid_argument("HogFilter: blockWidth must be greater than zero");
	if (blockHeight <= 0)
		throw invalid_argument("HogFilter: blockHeight must be greater than zero");
	if (signedAndUnsigned && binCount % 2 != 0)
		throw invalid_argument("HogFilter: the bin size must be even for signed and unsigned gradients to be combined");
}

Mat HogFilter::applyTo(const Mat& image, Mat& filtered) const {
	int cellRowCount = cvRound(static_cast<double>(image.rows) / static_cast<double>(cellHeight));
	int cellColumnCount = cvRound(static_cast<double>(image.cols) / static_cast<double>(cellWidth));
	Mat cellHistograms, cellEnergies;
	createCellHistograms(image, cellHistograms, binCount, cellRowCount, cellColumnCount, interpolate);
	computeCellEnergies(cellHistograms, cellEnergies, binCount, cellRowCount, cellColumnCount, signedAndUnsigned);
	createBlockHistograms(cellHistograms, cellEnergies, filtered, binCount,
			cellRowCount, cellColumnCount, blockWidth, blockHeight, signedAndUnsigned);
	return filtered;
}

void HogFilter::createBlockHistograms(const Mat& cellHistograms, const Mat& cellEnergies, Mat& blockHistograms,
		int binCount, int cellRowCount, int cellColumnCount, int blockWidth, int blockHeight, bool signedAndUnsigned) const {
	int binHalfCount = binCount / 2;
	int blockHistogramSize = signedAndUnsigned ? blockWidth * blockHeight * (binCount + binHalfCount) : blockWidth * blockHeight * binCount;
	int blockRowCount = cellRowCount - blockHeight + 1;
	int blockColumnCount = cellColumnCount - blockWidth + 1;
	blockHistograms.create(blockRowCount, blockColumnCount, CV_32FC(blockHistogramSize));
	float* blockHistogramValues = blockHistograms.ptr<float>();
	for (int blockRow = 0; blockRow < blockRowCount; ++blockRow) {
		for (int blockCol = 0; blockCol < blockColumnCount; ++blockCol) {
			float energy = 0;
			for (int cellRow = blockRow; cellRow < blockRow + blockHeight; ++cellRow) {
				for (int cellCol = blockCol; cellCol < blockCol + blockWidth; ++cellCol)
					energy += cellEnergies.at<float>(cellRow, cellCol);
			}
			float normalizer = 1.f / sqrt(energy + eps);
			for (int cellRow = blockRow; cellRow < blockRow + blockHeight; ++cellRow) {
				for (int cellCol = blockCol; cellCol < blockCol + blockWidth; ++cellCol) {
					const float* cellHistogramValues = cellHistograms.ptr<float>(cellRow, cellCol);
					for (int binIndex = 0; binIndex < binCount; ++binIndex)
						blockHistogramValues[binIndex] = normalizer * cellHistogramValues[binIndex];
					blockHistogramValues += binCount;
					if (signedAndUnsigned) { // sum up opposing gradients to get unsigned gradients and add to histogram
						for (int binIndex = 0; binIndex < binHalfCount; ++binIndex)
							blockHistogramValues[binIndex] = normalizer * (cellHistogramValues[binIndex] + cellHistogramValues[binHalfCount + binIndex]);
						blockHistogramValues += binHalfCount;
					}
				}
			}
		}
	}
}

void HogFilter::computeCellEnergies(const Mat& cellHistograms, Mat& cellEnergies,
		int binCount, int cellRowCount, int cellColumnCount, bool signedAndUnsigned) const {
	cellEnergies.create(cellRowCount, cellColumnCount, CV_32F);
	float* energyValues = cellEnergies.ptr<float>();
	const float* cellHistogramValues = cellHistograms.ptr<float>();
	int binHalfCount = binCount / 2;
	for (int cellIndex = 0; cellIndex < cellRowCount * cellColumnCount; ++cellIndex) {
		float energy = 0;
		if (signedAndUnsigned) {
			for (int binIndex = 0; binIndex < binHalfCount; ++binIndex) {
				float unsignedBinWeight = cellHistogramValues[binIndex] + cellHistogramValues[binHalfCount + binIndex];
				energy += unsignedBinWeight * unsignedBinWeight;
			}
		} else {
			for (int binIndex = 0; binIndex < binCount; ++binIndex)
				energy += cellHistogramValues[binIndex] * cellHistogramValues[binIndex];
		}
		energyValues[cellIndex] = energy;
		cellHistogramValues += binCount;
	}
}

} /* namespace imageprocessing */
