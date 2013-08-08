/*
 * SpatialHistogramFilter.cpp
 *
 *  Created on: 30.05.2013
 *      Author: poschmann
 */

#include "imageprocessing/SpatialHistogramFilter.hpp"
#include <stdexcept>

using cv::Vec2b;
using cv::Vec4b;
using std::runtime_error;

namespace imageprocessing {

SpatialHistogramFilter::SpatialHistogramFilter(
		unsigned int bins, int cellSize, int blockSize,
		bool interpolation, bool combineHistograms, Normalization normalization) :
				HistogramFilter(normalization),
				bins(bins),
				cellWidth(cellSize),
				cellHeight(cellSize),
				blockWidth(blockSize),
				blockHeight(blockSize),
				interpolation(interpolation),
				combineHistograms(combineHistograms),
				rowCache(),
				colCache() {}

SpatialHistogramFilter::SpatialHistogramFilter(
		unsigned int bins, int cellWidth, int cellHeight, int blockWidth, int blockHeight,
		bool interpolation, bool combineHistograms, Normalization normalization) :
				HistogramFilter(normalization),
				bins(bins),
				cellWidth(cellWidth),
				cellHeight(cellHeight),
				blockWidth(blockWidth),
				blockHeight(blockHeight),
				interpolation(interpolation),
				combineHistograms(combineHistograms),
				rowCache(),
				colCache() {}

SpatialHistogramFilter::~SpatialHistogramFilter() {}

Mat SpatialHistogramFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.channels() != 1 && image.channels() != 2 && image.channels() != 4)
		throw runtime_error("SpatialHistogramFilter: image must have one, two or four channels");
	if (image.depth() != CV_8U)
		throw runtime_error("SpatialHistogramFilter: image must have a depth of CV_8U");

	// create histograms over cells
	int cellRows = cvRound(static_cast<double>(image.rows) / static_cast<double>(cellHeight));
	int cellCols = cvRound(static_cast<double>(image.cols) / static_cast<double>(cellWidth));
	Mat cellHistograms = Mat::zeros(1, cellRows * cellCols * bins, CV_32F);
	float* cellHistogramsValues = cellHistograms.ptr<float>();

	float factor = 1.f / 255.f;
	if (interpolation) { // bilinear interpolation between cells
		createCache(rowCache, image.rows, cellRows);
		createCache(colCache, image.cols, cellCols);
		if (image.channels() == 1) { // bin information only, no weights
			for (int row = 0; row < image.rows; ++row) {
				const uchar* rowValues = image.ptr<uchar>(row);
				int cellRow0 = rowCache[row].index;
				int cellRow1 = cellRow0 + 1;
				float rowWeight0 = rowCache[row].weight;
				float rowWeight1 = 1.f - rowWeight0;
				for (int col = 0; col < image.cols; ++col) {
					uchar bin = rowValues[col];

					int cellCol0 = colCache[col].index;
					int cellCol1 = cellCol0 + 1;
					float colWeight0 = colCache[col].weight;
					float colWeight1 = 1.f - colWeight0;
					if (cellCol0 >= 0 && cellRow0 >= 0) {
						float* cellHistogramValues = cellHistogramsValues + cellRow0 * cellCols * bins + cellCol0 * bins;
						cellHistogramValues[bin] += colWeight1 * rowWeight1;
					}
					if (cellCol1 < cellCols && cellRow0 >= 0) {
						float* cellHistogramValues = cellHistogramsValues + cellRow0 * cellCols * bins + cellCol1 * bins;
						cellHistogramValues[bin] += colWeight0 * rowWeight1;
					}
					if (cellCol0 >= 0 && cellRow1 < cellRows) {
						float* cellHistogramValues = cellHistogramsValues + cellRow1 * cellCols * bins + cellCol0 * bins;
						cellHistogramValues[bin] += colWeight1 * rowWeight0;
					}
					if (cellCol1 < cellCols && cellRow1 < cellRows) {
						float* cellHistogramValues = cellHistogramsValues + cellRow1 * cellCols * bins + cellCol1 * bins;
						cellHistogramValues[bin] += colWeight0 * rowWeight0;
					}
				}
			}
		} else if (image.channels() == 2) { // bin index and weight available
			for (int row = 0; row < image.rows; ++row) {
				const Vec2b* rowValues = image.ptr<Vec2b>(row);
				int cellRow0 = rowCache[row].index;
				int cellRow1 = cellRow0 + 1;
				float rowWeight0 = rowCache[row].weight;
				float rowWeight1 = 1.f - rowWeight0;
				for (int col = 0; col < image.cols; ++col) {
					uchar bin = rowValues[col][0];
					float weight = factor * rowValues[col][1];

					int cellCol0 = colCache[col].index;
					int cellCol1 = cellCol0 + 1;
					float colWeight0 = colCache[col].weight;
					float colWeight1 = 1.f - colWeight0;
					if (cellCol0 >= 0 && cellRow0 >= 0) {
						float* cellHistogramValues = cellHistogramsValues + cellRow0 * cellCols * bins + cellCol0 * bins;
						cellHistogramValues[bin] += weight * colWeight1 * rowWeight1;
					}
					if (cellCol1 < cellCols && cellRow0 >= 0) {
						float* cellHistogramValues = cellHistogramsValues + cellRow0 * cellCols * bins + cellCol1 * bins;
						cellHistogramValues[bin] += weight * colWeight0 * rowWeight1;
					}
					if (cellCol0 >= 0 && cellRow1 < cellRows) {
						float* cellHistogramValues = cellHistogramsValues + cellRow1 * cellCols * bins + cellCol0 * bins;
						cellHistogramValues[bin] += weight * colWeight1 * rowWeight0;
					}
					if (cellCol1 < cellCols && cellRow1 < cellRows) {
						float* cellHistogramValues = cellHistogramsValues + cellRow1 * cellCols * bins + cellCol1 * bins;
						cellHistogramValues[bin] += weight * colWeight0 * rowWeight0;
					}
				}
			}
		} else if (image.channels() == 4) { // two bin indices and weights available
			for (int row = 0; row < image.rows; ++row) {
				const Vec4b* rowValues = image.ptr<Vec4b>(row);
				int cellRow0 = rowCache[row].index;
				int cellRow1 = cellRow0 + 1;
				float rowWeight0 = rowCache[row].weight;
				float rowWeight1 = 1.f - rowWeight0;
				for (int col = 0; col < image.cols; ++col) {
					uchar bin1 = rowValues[col][0];
					float weight1 = factor * rowValues[col][1];
					uchar bin2 = rowValues[col][2];
					float weight2 = factor * rowValues[col][3];

					int cellCol0 = colCache[col].index;
					int cellCol1 = cellCol0 + 1;
					float colWeight0 = colCache[col].weight;
					float colWeight1 = 1.f - colWeight0;
					if (cellCol0 >= 0 && cellRow0 >= 0) {
						float* cellHistogramValues = cellHistogramsValues + cellRow0 * cellCols * bins + cellCol0 * bins;
						cellHistogramValues[bin1] += weight1 * colWeight1 * rowWeight1;
						cellHistogramValues[bin2] += weight2 * colWeight1 * rowWeight1;
					}
					if (cellCol1 < cellCols && cellRow0 >= 0) {
						float* cellHistogramValues = cellHistogramsValues + cellRow0 * cellCols * bins + cellCol1 * bins;
						cellHistogramValues[bin1] += weight1 * colWeight0 * rowWeight1;
						cellHistogramValues[bin2] += weight2 * colWeight0 * rowWeight1;
					}
					if (cellCol0 >= 0 && cellRow1 < cellRows) {
						float* cellHistogramValues = cellHistogramsValues + cellRow1 * cellCols * bins + cellCol0 * bins;
						cellHistogramValues[bin1] += weight1 * colWeight1 * rowWeight0;
						cellHistogramValues[bin2] += weight2 * colWeight1 * rowWeight0;
					}
					if (cellCol1 < cellCols && cellRow1 < cellRows) {
						float* cellHistogramValues = cellHistogramsValues + cellRow1 * cellCols * bins + cellCol1 * bins;
						cellHistogramValues[bin1] += weight1 * colWeight0 * rowWeight0;
						cellHistogramValues[bin2] += weight2 * colWeight0 * rowWeight0;
					}
				}
			}
		}
	} else { // no bilinear interpolation between cells
		if (image.channels() == 1) { // bin information only, no weights
			for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
				for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
					float* histogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
					int startRow = (cellRow * image.rows) / cellRows;
					int startCol = (cellCol * image.cols) / cellCols;
					int endRow = ((cellRow + 1) * image.rows) / cellRows;
					int endCol = ((cellCol + 1) * image.cols) / cellCols;
					for (int row = startRow; row < endRow; ++row) {
						const uchar* rowValues = image.ptr<uchar>(row);
						for (int col = startCol; col < endCol; ++col)
							histogramValues[rowValues[col]]++;
					}
				}
			}
		} else if (image.channels() == 2) { // bin index and weight available
			for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
				for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
					float* histogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
					int startRow = (cellRow * image.rows) / cellRows;
					int startCol = (cellCol * image.cols) / cellCols;
					int endRow = ((cellRow + 1) * image.rows) / cellRows;
					int endCol = ((cellCol + 1) * image.cols) / cellCols;
					for (int row = startRow; row < endRow; ++row) {
						const Vec2b* rowValues = image.ptr<Vec2b>(row);
						for (int col = startCol; col < endCol; ++col) {
							uchar bin = rowValues[col][0];
							uchar weight = rowValues[col][1];
							histogramValues[bin] += factor * weight;
						}
					}
				}
			}
		} else if (image.channels() == 4) { // two bin indices and weights available
			for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
				for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
					float* histogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
					int startRow = (cellRow * image.rows) / cellRows;
					int startCol = (cellCol * image.cols) / cellCols;
					int endRow = ((cellRow + 1) * image.rows) / cellRows;
					int endCol = ((cellCol + 1) * image.cols) / cellCols;
					for (int row = startRow; row < endRow; ++row) {
						const Vec4b* rowValues = image.ptr<Vec4b>(row);
						for (int col = startCol; col < endCol; ++col) {
							uchar bin1 = rowValues[col][0];
							uchar weight1 = rowValues[col][1];
							uchar bin2 = rowValues[col][2];
							uchar weight2 = rowValues[col][3];
							histogramValues[bin1] += factor * weight1;
							histogramValues[bin2] += factor * weight2;
						}
					}
				}
			}
		}
	}

	// create histograms over blocks
	int blockHistogramSize = combineHistograms ? bins : blockWidth * blockHeight * bins;
	int blockRows = cellRows - blockHeight + 1;
	int blockCols = cellCols - blockWidth + 1;
	filtered.create(1, blockRows * blockCols * blockHistogramSize, CV_32F);
	for (int blockRow = 0; blockRow < blockRows; ++blockRow) {
		for (int blockCol = 0; blockCol < blockCols; ++blockCol) {
			Mat blockHistogram(filtered,
					cv::Range(0, 1),
					cv::Range((blockRow * blockCols + blockCol) * blockHistogramSize, (blockRow * blockCols + blockCol + 1) * blockHistogramSize));
			float* blockHistogramValues = blockHistogram.ptr<float>();
			if (combineHistograms) { // combine histograms by adding the bin values
				blockHistogram = Mat::zeros(1, bins, CV_32F);
				for (int cellRow = blockRow; cellRow < blockRow + blockHeight; ++cellRow) {
					for (int cellCol = blockCol; cellCol < blockCol + blockWidth; ++cellCol) {
						float* cellHistogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
						for (unsigned int bin = 0; bin < bins; ++bin)
							blockHistogramValues[bin] += cellHistogramValues[bin];
					}
				}
			} else { // create concatenation of histograms
				for (int cellRow = blockRow; cellRow < blockRow + blockHeight; ++cellRow) {
					for (int cellCol = blockCol; cellCol < blockCol + blockWidth; ++cellCol) {
						float* cellHistogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
						for (unsigned int bin = 0; bin < bins; ++bin)
							blockHistogramValues[bin] = cellHistogramValues[bin];
						blockHistogramValues += bins;
					}
				}
			}
			normalize(blockHistogram);
		}
	}
	return filtered;
}

void SpatialHistogramFilter::createCache(vector<CacheEntry>& cache, unsigned int size, int count) const {
	if (cache.size() != size) {
		cache.clear();
		cache.reserve(size);
		CacheEntry entry;
		for (unsigned int matIndex = 0; matIndex < size; ++matIndex) {
			double realIndex = static_cast<double>(count) * (static_cast<double>(matIndex) + 0.5) / static_cast<double>(size) - 0.5;
			entry.index = static_cast<int>(floor(realIndex));
			entry.weight = realIndex - entry.index;
			cache.push_back(entry);
		}
	}
}

} /* namespace imageprocessing */
