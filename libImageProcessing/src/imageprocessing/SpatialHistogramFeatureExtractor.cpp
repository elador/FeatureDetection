/*
 * SpatialHistogramFeatureExtractor.cpp
 *
 *  Created on: 30.05.2013
 *      Author: poschmann
 */

#include "imageprocessing/SpatialHistogramFeatureExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include <stdexcept>

using cv::Vec2b;
using cv::Vec4b;
using std::runtime_error;

namespace imageprocessing {

SpatialHistogramFeatureExtractor::SpatialHistogramFeatureExtractor(shared_ptr<FeatureExtractor> extractor,
		unsigned int bins, int cellSize, int blockSize,
		bool interpolation, bool combineHistograms, Normalization normalization) :
				HistogramFeatureExtractor(normalization),
				extractor(extractor),
				bins(bins),
				cellWidth(cellSize),
				cellHeight(cellSize),
				blockWidth(blockSize),
				blockHeight(blockSize),
				interpolation(interpolation),
				combineHistograms(combineHistograms),
				rowCache(),
				colCache() {}

SpatialHistogramFeatureExtractor::SpatialHistogramFeatureExtractor(
		shared_ptr<FeatureExtractor> extractor, unsigned int bins,
		int cellWidth, int cellHeight, int blockWidth, int blockHeight,
		bool interpolation, bool combineHistograms, Normalization normalization) :
				HistogramFeatureExtractor(normalization),
				extractor(extractor),
				bins(bins),
				cellWidth(cellWidth),
				cellHeight(cellHeight),
				blockWidth(blockWidth),
				blockHeight(blockHeight),
				interpolation(interpolation),
				combineHistograms(combineHistograms),
				rowCache(),
				colCache() {}

SpatialHistogramFeatureExtractor::~SpatialHistogramFeatureExtractor() {}

void SpatialHistogramFeatureExtractor::update(const Mat& image) {
	extractor->update(image);
}

void SpatialHistogramFeatureExtractor::update(shared_ptr<VersionedImage> image) {
	extractor->update(image);
}

shared_ptr<Patch> SpatialHistogramFeatureExtractor::extract(int x, int y, int width, int height) const {
	shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
	if (patch) {
		Mat& patchData = patch->getData();
		if (patchData.channels() == 3)
			throw runtime_error("SpatialHistogramFeatureExtractor: Patch data must have one, two or four channels");
		if (patchData.depth() != CV_8U)
			throw runtime_error("SpatialHistogramFeatureExtractor: Patch data must have a depth of CV_8U");
		// create histograms of cells
		int cellRows = cvRound(static_cast<double>(patchData.rows) / static_cast<double>(cellHeight));
		int cellCols = cvRound(static_cast<double>(patchData.cols) / static_cast<double>(cellWidth));
		Mat cellHistograms = Mat::zeros(1, cellRows * cellCols * bins, CV_32F);
		float* cellHistogramsValues = cellHistograms.ptr<float>();

		float factor = 1.f / 255.f;
		if (interpolation) { // bilinear interpolation between cells
			createCache(rowCache, patchData.rows, cellHeight);
			createCache(colCache, patchData.cols, cellWidth);
			if (patchData.channels() == 1) { // bin information only, no weights
				for (int row = 0; row < patchData.rows; ++row) {
					uchar* rowValues = patchData.ptr<uchar>(row);
					int cellRow0 = rowCache[row].index;
					int cellRow1 = cellRow0 + 1;
					float rowWeight0 = rowCache[row].weight;
					float rowWeight1 = 1.f - rowWeight0;
					for (int col = 0; col < patchData.cols; ++col) {
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
			} else if (patchData.channels() == 2) { // bin index and weight available
				for (int row = 0; row < patchData.rows; ++row) {
					Vec2b* rowValues = patchData.ptr<Vec2b>(row);
					int cellRow0 = rowCache[row].index;
					int cellRow1 = cellRow0 + 1;
					float rowWeight0 = rowCache[row].weight;
					float rowWeight1 = 1.f - rowWeight0;
					for (int col = 0; col < patchData.cols; ++col) {
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
			} else if (patchData.channels() == 4) { // two bin indices and weights available
				for (int row = 0; row < patchData.rows; ++row) {
					Vec4b* rowValues = patchData.ptr<Vec4b>(row);
					int cellRow0 = rowCache[row].index;
					int cellRow1 = cellRow0 + 1;
					float rowWeight0 = rowCache[row].weight;
					float rowWeight1 = 1.f - rowWeight0;
					for (int col = 0; col < patchData.cols; ++col) {
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
			if (patchData.channels() == 1) { // bin information only, no weights
				for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
					for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
						float* histogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
						int startRow = (cellRow * patchData.rows) / cellRows;
						int startCol = (cellCol * patchData.cols) / cellCols;
						int endRow = ((cellRow + 1) * patchData.rows) / cellRows;
						int endCol = ((cellCol + 1) * patchData.cols) / cellCols;
						for (int row = startRow; row < endRow; ++row) {
							uchar* rowValues = patchData.ptr<uchar>(row);
							for (int col = startCol; col < endCol; ++col)
								histogramValues[rowValues[col]]++;
						}
					}
				}
			} else if (patchData.channels() == 2) { // bin index and weight available
				for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
					for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
						float* histogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
						int startRow = (cellRow * patchData.rows) / cellRows;
						int startCol = (cellCol * patchData.cols) / cellCols;
						int endRow = ((cellRow + 1) * patchData.rows) / cellRows;
						int endCol = ((cellCol + 1) * patchData.cols) / cellCols;
						for (int row = startRow; row < endRow; ++row) {
							Vec2b* rowValues = patchData.ptr<Vec2b>(row);
							for (int col = startCol; col < endCol; ++col) {
								uchar bin = rowValues[col][0];
								uchar weight = rowValues[col][1];
								histogramValues[bin] += factor * weight;
							}
						}
					}
				}
			} else if (patchData.channels() == 4) { // two bin indices and weights available
				for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
					for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
						float* histogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
						int startRow = (cellRow * patchData.rows) / cellRows;
						int startCol = (cellCol * patchData.cols) / cellCols;
						int endRow = ((cellRow + 1) * patchData.rows) / cellRows;
						int endCol = ((cellCol + 1) * patchData.cols) / cellCols;
						for (int row = startRow; row < endRow; ++row) {
							Vec4b* rowValues = patchData.ptr<Vec4b>(row);
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

		// create histograms of blocks
		int blockHistogramSize = combineHistograms ? bins : blockWidth * blockHeight * bins;
		int blockRows = cellRows - blockHeight + 1;
		int blockCols = cellCols - blockWidth + 1;
		vector<Mat> blockHistograms(blockRows * blockCols);
		for (int blockRow = 0; blockRow < blockRows; ++blockRow) {
			for (int blockCol = 0; blockCol < blockCols; ++blockCol) {
				Mat& blockHistogram = blockHistograms[blockRow * blockCols + blockCol];
				if (combineHistograms) { // combine histograms by adding the bin values
					blockHistogram = Mat::zeros(1, bins, CV_32F);
					float* blockHistogramValues = blockHistogram.ptr<float>(0);
					for (int cellRow = blockRow; cellRow < blockRow + blockHeight; ++cellRow) {
						for (int cellCol = blockCol; cellCol < blockCol + blockWidth; ++cellCol) {
							float* cellHistogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
							for (unsigned int bin = 0; bin < bins; ++bin)
								blockHistogramValues[bin] += cellHistogramValues[bin];
						}
					}
				} else { // create concatenation of histograms
					blockHistogram.create(1, blockHistogramSize, CV_32F);
					float* blockHistogramValues = blockHistogram.ptr<float>(0);
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
		patchData.create(1, blockHistograms.size() * blockHistogramSize, CV_32F);
		float* patchValues = patchData.ptr<float>(0);
		for (unsigned int histIndex = 0; histIndex < blockHistograms.size(); ++histIndex) {
			Mat& blockHistogram = blockHistograms[histIndex];
			float* blockHistogramValues = blockHistogram.ptr<float>(0);
			for (int bin = 0; bin < blockHistogramSize; ++bin)
				patchValues[bin] = blockHistogramValues[bin];
			patchValues += blockHistogramSize;
		}
	}
	return patch;
}

void SpatialHistogramFeatureExtractor::createCache(vector<CacheEntry>& cache, unsigned int size, int cellSize) const {
	if (cache.size() != size) {
		cache.clear();
		cache.reserve(size);
		CacheEntry entry;
		for (unsigned int matIndex = 0; matIndex < size; ++matIndex) {
			double realIndex = (static_cast<double>(matIndex) + 0.5) / static_cast<double>(cellSize) - 0.5;
			entry.index = static_cast<int>(floor(realIndex));
			entry.weight = realIndex - entry.index;
			cache.push_back(entry);
		}
	}
}

} /* namespace imageprocessing */
