/*
 * SpatialHistogramFilter.cpp
 *
 *  Created on: 30.05.2013
 *      Author: poschmann
 */

#include "imageprocessing/SpatialHistogramFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;

namespace imageprocessing {

SpatialHistogramFilter::SpatialHistogramFilter(int binCount, int cellSize, int blockSize,
		bool interpolate, bool concatenate, Normalization normalization) :
				HistogramFilter(normalization),
				binCount(binCount),
				cellWidth(cellSize),
				cellHeight(cellSize),
				blockWidth(blockSize),
				blockHeight(blockSize),
				interpolate(interpolate),
				concatenate(concatenate) {
	if (binCount <= 0)
		throw invalid_argument("SpatialHistogramFilter: binCount must be greater than zero");
	if (cellSize <= 0)
		throw invalid_argument("SpatialHistogramFilter: cellSize must be greater than zero");
	if (blockSize <= 0)
		throw invalid_argument("SpatialHistogramFilter: blockSize must be greater than zero");
}

SpatialHistogramFilter::SpatialHistogramFilter(int binCount, int cellWidth, int cellHeight, int blockWidth, int blockHeight,
		bool interpolate, bool concatenate, Normalization normalization) :
				HistogramFilter(normalization),
				binCount(binCount),
				cellWidth(cellWidth),
				cellHeight(cellHeight),
				blockWidth(blockWidth),
				blockHeight(blockHeight),
				interpolate(interpolate),
				concatenate(concatenate) {
	if (binCount <= 0)
		throw invalid_argument("SpatialHistogramFilter: binCount must be greater than zero");
	if (cellWidth <= 0)
		throw invalid_argument("SpatialHistogramFilter: cellWidth must be greater than zero");
	if (cellHeight <= 0)
		throw invalid_argument("SpatialHistogramFilter: cellHeight must be greater than zero");
	if (blockWidth <= 0)
		throw invalid_argument("SpatialHistogramFilter: blockWidth must be greater than zero");
	if (blockHeight <= 0)
		throw invalid_argument("SpatialHistogramFilter: blockHeight must be greater than zero");
}

Mat SpatialHistogramFilter::applyTo(const Mat& image, Mat& filtered) const {
	int cellRowCount = cvRound(static_cast<double>(image.rows) / static_cast<double>(cellHeight));
	int cellColumnCount = cvRound(static_cast<double>(image.cols) / static_cast<double>(cellWidth));
	if (blockWidth == 1 && blockHeight == 1) { // cell histograms only, normalize over whole concatenated histogram
		createCellHistograms(image, filtered, binCount, cellRowCount, cellColumnCount, interpolate);
		normalize(filtered);
	} else { // build blocks using cells, normalize over blocks
		Mat cellHistograms;
		createCellHistograms(image, cellHistograms, binCount, cellRowCount, cellColumnCount, interpolate);
		createBlockHistograms(cellHistograms, filtered, binCount, cellRowCount, cellColumnCount, blockWidth, blockHeight, concatenate);
	}
	return filtered;
}

void SpatialHistogramFilter::createBlockHistograms(const Mat& cellHistograms, Mat& blockHistograms, int binCount,
		int cellRowCount, int cellColumnCount, int blockWidth, int blockHeight, bool concatenate) const {
	int blockHistogramSize = concatenate ? blockWidth * blockHeight * binCount : binCount;
	int blockRowCount = cellRowCount - blockHeight + 1;
	int blockColumnCount = cellColumnCount - blockWidth + 1;
	blockHistograms = Mat::zeros(blockRowCount, blockColumnCount, CV_32FC(blockHistogramSize));
	float* blockHistogramValues = blockHistograms.ptr<float>();
	for (int blockRow = 0; blockRow < blockRowCount; ++blockRow) {
		for (int blockCol = 0; blockCol < blockColumnCount; ++blockCol) {
			Mat blockHistogram(1, blockHistogramSize, CV_32F, blockHistogramValues);
			for (int cellRow = blockRow; cellRow < blockRow + blockHeight; ++cellRow) {
				for (int cellCol = blockCol; cellCol < blockCol + blockWidth; ++cellCol) {
					const float* cellHistogramValues = cellHistograms.ptr<float>(cellRow, cellCol);
					for (int bin = 0; bin < binCount; ++bin)
						blockHistogramValues[bin] += cellHistogramValues[bin];
					if (concatenate) // if histograms should be concatenated,
						blockHistogramValues += binCount; // then the insertion point of the next values has to be shifted
				}
			}
			if (!concatenate) // if the histograms should not be concatenated,
				blockHistogramValues += binCount; // then the insertion point must now be shifted to the next histogram (was not shifted before)
			normalize(blockHistogram);
		}
	}
}

} /* namespace imageprocessing */
