/*
 * ExtendedHogExtractor.cpp
 *
 *  Created on: 01.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/ExtendedHogExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include <stdexcept>

using cv::Vec2b;
using cv::Vec4b;
using std::runtime_error;

namespace imageprocessing {

const float ExtendedHogExtractor::eps = 0.0001;

ExtendedHogExtractor::ExtendedHogExtractor(shared_ptr<FeatureExtractor> extractor,
		unsigned int bins, int cellSize, bool interpolation, bool signedAndUnsigned, float alpha) :
				extractor(extractor),
				bins(bins),
				cellWidth(cellSize),
				cellHeight(cellSize),
				interpolation(interpolation),
				signedAndUnsigned(signedAndUnsigned),
				alpha(alpha),
				rowCache(),
				colCache() {}

ExtendedHogExtractor::ExtendedHogExtractor(
		shared_ptr<FeatureExtractor> extractor, unsigned int bins,
		int cellWidth, int cellHeight, bool interpolation, bool signedAndUnsigned, float alpha) :
				extractor(extractor),
				bins(bins),
				cellWidth(cellWidth),
				cellHeight(cellHeight),
				interpolation(interpolation),
				signedAndUnsigned(signedAndUnsigned),
				alpha(alpha),
				rowCache(),
				colCache() {}

ExtendedHogExtractor::~ExtendedHogExtractor() {}

void ExtendedHogExtractor::update(const Mat& image) {
	extractor->update(image);
}

void ExtendedHogExtractor::update(shared_ptr<VersionedImage> image) {
	extractor->update(image);
}

shared_ptr<Patch> ExtendedHogExtractor::extract(int x, int y, int width, int height) const {
	shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
	if (patch) {
		Mat& patchData = patch->getData();
		if (patchData.channels() != 2 && patchData.channels() != 4)
			throw runtime_error("ExtendedHogExtractor: patch data must have two or four channels");
		if (patchData.depth() != CV_8U)
			throw runtime_error("ExtendedHogExtractor: patch data must have a depth of CV_8U");

		// create histograms over cells and gradient energy of cells
		int cellRows = cvRound(static_cast<double>(patchData.rows) / static_cast<double>(cellHeight));
		int cellCols = cvRound(static_cast<double>(patchData.cols) / static_cast<double>(cellWidth));
		Mat cellHistograms = Mat::zeros(1, cellRows * cellCols * bins, CV_32F);
		Mat cellEnergies = Mat::zeros(1, cellRows * cellCols, CV_32F);
		float* cellHistogramsValues = cellHistograms.ptr<float>();
		float* cellEnergiesValues = cellEnergies.ptr<float>();

		// compute cell histograms
		float factor = 1.f / 255.f;
		if (interpolation) { // bilinear interpolation between cells
			createCache(rowCache, patchData.rows, cellRows);
			createCache(colCache, patchData.cols, cellCols);
			if (patchData.channels() == 2) { // bin index and weight available
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
			if (patchData.channels() == 2) { // bin index and weight available
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

		// create extended HOG feature vector
		if (signedAndUnsigned) { // signed and unsigned gradients should be combined into descriptor
			if (bins % 2 != 0)
				throw new runtime_error("ExtendedHogExtractor: the bin size must be even for signed and unsigned gradients to be combined");

			// compute gradient energy over cells
			unsigned int halfBins = bins / 2;
			for (int cellIndex = 0; cellIndex < cellRows * cellCols; ++cellIndex) {
				float* histogramValues = cellHistogramsValues + cellIndex * bins;
				for (unsigned int bin = 0; bin < halfBins; ++bin) {
					float sum = histogramValues[bin] + histogramValues[bin + halfBins];
					cellEnergiesValues[cellIndex] += sum * sum;
				}
			}

			// create descriptors
			patchData.create(1, (cellRows - 2) * (cellCols - 2) * (bins + halfBins + 4), CV_32F);
			float* patchValues = patchData.ptr<float>(0);
			for (int cellRow = 1; cellRow < cellRows - 1; ++cellRow) {
				for (int cellCol = 1; cellCol < cellCols - 1; ++cellCol) {
					float* cellHistogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
					float sqn00 = cellEnergiesValues[(cellRow - 1) * cellCols + (cellCol - 1)];
					float sqn01 = cellEnergiesValues[(cellRow - 1) * cellCols + (cellCol)];
					float sqn02 = cellEnergiesValues[(cellRow - 1) * cellCols + (cellCol + 1)];
					float sqn10 = cellEnergiesValues[(cellRow) * cellCols + (cellCol - 1)];
					float sqn11 = cellEnergiesValues[(cellRow) * cellCols + (cellCol)];
					float sqn12 = cellEnergiesValues[(cellRow) * cellCols + (cellCol + 1)];
					float sqn20 = cellEnergiesValues[(cellRow + 1) * cellCols + (cellCol - 1)];
					float sqn21 = cellEnergiesValues[(cellRow + 1) * cellCols + (cellCol)];
					float sqn22 = cellEnergiesValues[(cellRow + 1) * cellCols + (cellCol + 1)];
					float n1 = 1.f / sqrt(sqn00 + sqn01 + sqn10 + sqn11 + eps);
					float n2 = 1.f / sqrt(sqn01 + sqn02 + sqn11 + sqn12 + eps);
					float n3 = 1.f / sqrt(sqn10 + sqn11 + sqn20 + sqn21 + eps);
					float n4 = 1.f / sqrt(sqn11 + sqn12 + sqn21 + sqn22 + eps);

					float t1 = 0;
					float t2 = 0;
					float t3 = 0;
					float t4 = 0;

					// signed orientation features (aka contrast-sensitive)
					for (unsigned int bin = 0; bin < bins; ++bin) {
						float h1 = std::min(alpha, cellHistogramValues[bin] * n1);
						float h2 = std::min(alpha, cellHistogramValues[bin] * n2);
						float h3 = std::min(alpha, cellHistogramValues[bin] * n3);
						float h4 = std::min(alpha, cellHistogramValues[bin] * n4);
						patchValues[bin] = 0.5 * (h1 + h2 + h3 + h4);
						t1 += h1;
						t2 += h2;
						t3 += h3;
						t4 += h4;
					}
					patchValues += bins;

					// unsigned orientation features (aka contrast-insensitive)
					for (unsigned int bin = 0; bin < halfBins; ++bin) {
						float sum = cellHistogramValues[bin] + cellHistogramValues[bin + halfBins];
						float h1 = std::min(alpha, sum * n1);
						float h2 = std::min(alpha, sum * n2);
						float h3 = std::min(alpha, sum * n3);
						float h4 = std::min(alpha, sum * n4);
						patchValues[bin] = 0.5 * (h1 + h2 + h3 + h4);
					}
					patchValues += halfBins;

					// energy features
					patchValues[0] = 0.2357 * t1;
					patchValues[1] = 0.2357 * t2;
					patchValues[2] = 0.2357 * t3;
					patchValues[3] = 0.2357 * t4;
					patchValues += 4;
				}
			}
		} else { // only signed or unsigned gradients should be in descriptor

			// compute gradient energy over cells
			for (int cellIndex = 0; cellIndex < cellRows * cellCols; ++cellIndex) {
				float* histogramValues = cellHistogramsValues + cellIndex * bins;
				for (unsigned int bin = 0; bin < bins; ++bin)
					cellEnergiesValues[cellIndex] += histogramValues[bin] * histogramValues[bin];
			}

			// create descriptors
			patchData.create(1, (cellRows - 2) * (cellCols - 2) * (bins + 4), CV_32F);
			float* patchValues = patchData.ptr<float>(0);
			for (int cellRow = 1; cellRow < cellRows - 1; ++cellRow) {
				for (int cellCol = 1; cellCol < cellCols - 1; ++cellCol) {
					float* cellHistogramValues = cellHistogramsValues + cellRow * cellCols * bins + cellCol * bins;
					float sqn00 = cellEnergiesValues[(cellRow - 1) * cellCols + (cellCol - 1)];
					float sqn01 = cellEnergiesValues[(cellRow - 1) * cellCols + (cellCol)];
					float sqn02 = cellEnergiesValues[(cellRow - 1) * cellCols + (cellCol + 1)];
					float sqn10 = cellEnergiesValues[(cellRow) * cellCols + (cellCol - 1)];
					float sqn11 = cellEnergiesValues[(cellRow) * cellCols + (cellCol)];
					float sqn12 = cellEnergiesValues[(cellRow) * cellCols + (cellCol + 1)];
					float sqn20 = cellEnergiesValues[(cellRow + 1) * cellCols + (cellCol - 1)];
					float sqn21 = cellEnergiesValues[(cellRow + 1) * cellCols + (cellCol)];
					float sqn22 = cellEnergiesValues[(cellRow + 1) * cellCols + (cellCol + 1)];
					float n1 = 1.f / sqrt(sqn00 + sqn01 + sqn10 + sqn11 + eps);
					float n2 = 1.f / sqrt(sqn01 + sqn02 + sqn11 + sqn12 + eps);
					float n3 = 1.f / sqrt(sqn10 + sqn11 + sqn20 + sqn21 + eps);
					float n4 = 1.f / sqrt(sqn11 + sqn12 + sqn21 + sqn22 + eps);

					float t1 = 0;
					float t2 = 0;
					float t3 = 0;
					float t4 = 0;

					// orientation features
					for (unsigned int bin = 0; bin < bins; ++bin) {
						float h1 = std::min(alpha, cellHistogramValues[bin] * n1);
						float h2 = std::min(alpha, cellHistogramValues[bin] * n2);
						float h3 = std::min(alpha, cellHistogramValues[bin] * n3);
						float h4 = std::min(alpha, cellHistogramValues[bin] * n4);
						patchValues[bin] = 0.5 * (h1 + h2 + h3 + h4);
						t1 += h1;
						t2 += h2;
						t3 += h3;
						t4 += h4;
					}

					// energy features
					patchValues += bins;
					patchValues[0] = 0.2357 * t1;
					patchValues[1] = 0.2357 * t2;
					patchValues[2] = 0.2357 * t3;
					patchValues[3] = 0.2357 * t4;
					patchValues += 4;
				}
			}
		}
	}
	return patch;
}

void ExtendedHogExtractor::createCache(vector<CacheEntry>& cache, unsigned int size, int count) const {
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
