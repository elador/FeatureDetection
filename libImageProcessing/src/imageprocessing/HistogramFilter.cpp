/*
 * HistogramFilter.cpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/HistogramFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Vec2b;
using cv::Vec4b;
using std::vector;
using std::runtime_error;

namespace imageprocessing {

const float HistogramFilter::eps = 1e-4;

HistogramFilter::HistogramFilter(Normalization normalization) : normalization(normalization),	rowCache(), colCache() {}

void HistogramFilter::createCellHistograms(const Mat& image, Mat& histograms, int binCount, int rowCount, int columnCount, bool interpolate) const {
	if (image.channels() != 1 && image.channels() != 2 && image.channels() != 4)
		throw runtime_error("HistogramFilter: image must have one, two or four channels");
	if (image.depth() != CV_8U)
		throw runtime_error("HistogramFilter: image must have a depth of CV_8U");

	histograms = Mat::zeros(rowCount, columnCount, CV_32FC(binCount));
	float factor = 1.f / 255.f;
	if (interpolate) { // bilinear interpolation between cells
		createCache(rowCache, image.rows, rowCount);
		createCache(colCache, image.cols, columnCount);
		if (image.channels() == 1) { // bin information only, no weights
			for (int imageRow = 0; imageRow < image.rows; ++imageRow) {
				const uchar* rowValues = image.ptr<uchar>(imageRow);
				int rowIndex0 = rowCache[imageRow].index1;
				int rowIndex1 = rowCache[imageRow].index2;
				float rowWeight1 = rowCache[imageRow].weight2;
				float rowWeight0 = rowCache[imageRow].weight1;
				for (int imageCol = 0; imageCol < image.cols; ++imageCol) {
					uchar bin = rowValues[imageCol];

					int colIndex0 = colCache[imageCol].index1;
					int colIndex1 = colCache[imageCol].index2;
					float colWeight1 = colCache[imageCol].weight2;
					float colWeight0 = colCache[imageCol].weight1;
					if (rowIndex0 >= 0 && colIndex0 >= 0) {
						float* histogramValues = histograms.ptr<float>(rowIndex0, colIndex0);
						histogramValues[bin] += rowWeight0 * colWeight0;
					}
					if (rowIndex0 >= 0 && colIndex1 < columnCount) {
						float* histogramValues = histograms.ptr<float>(rowIndex0, colIndex1);
						histogramValues[bin] += rowWeight0 * colWeight1;
					}
					if (rowIndex1 < rowCount && colIndex0 >= 0) {
						float* histogramValues = histograms.ptr<float>(rowIndex1, colIndex0);
						histogramValues[bin] += rowWeight1 * colWeight0;
					}
					if (rowIndex1 < rowCount && colIndex1 < columnCount) {
						float* histogramValues = histograms.ptr<float>(rowIndex1, colIndex1);
						histogramValues[bin] += rowWeight1 * colWeight1;
					}
				}
			}
		} else if (image.channels() == 2) { // bin index and weight available
			for (int imageRow = 0; imageRow < image.rows; ++imageRow) {
				const Vec2b* rowValues = image.ptr<Vec2b>(imageRow);
				int rowIndex0 = rowCache[imageRow].index1;
				int rowIndex1 = rowCache[imageRow].index2;
				float rowWeight1 = rowCache[imageRow].weight2;
				float rowWeight0 = rowCache[imageRow].weight1;
				for (int imageCol = 0; imageCol < image.cols; ++imageCol) {
					uchar bin = rowValues[imageCol][0];
					float weight = factor * rowValues[imageCol][1];

					int colIndex0 = colCache[imageCol].index1;
					int colIndex1 = colCache[imageCol].index2;
					float colWeight1 = colCache[imageCol].weight2;
					float colWeight0 = colCache[imageCol].weight1;
					if (rowIndex0 >= 0 && colIndex0 >= 0) {
						float* histogramValues = histograms.ptr<float>(rowIndex0, colIndex0);
						histogramValues[bin] += weight * rowWeight0 * colWeight0;
					}
					if (rowIndex0 >= 0 && colIndex1 < columnCount) {
						float* histogramValues = histograms.ptr<float>(rowIndex0, colIndex1);
						histogramValues[bin] += weight * rowWeight0 * colWeight1;
					}
					if (rowIndex1 < rowCount && colIndex0 >= 0) {
						float* histogramValues = histograms.ptr<float>(rowIndex1, colIndex0);
						histogramValues[bin] += weight * rowWeight1 * colWeight0;
					}
					if (rowIndex1 < rowCount && colIndex1 < columnCount) {
						float* histogramValues = histograms.ptr<float>(rowIndex1, colIndex1);
						histogramValues[bin] += weight * rowWeight1 * colWeight1;
					}
				}
			}
		} else if (image.channels() == 4) { // two bin indices and weights available
			for (int imageRow = 0; imageRow < image.rows; ++imageRow) {
				const Vec4b* rowValues = image.ptr<Vec4b>(imageRow);
				int rowIndex0 = rowCache[imageRow].index1;
				int rowIndex1 = rowCache[imageRow].index2;
				float rowWeight1 = rowCache[imageRow].weight2;
				float rowWeight0 = rowCache[imageRow].weight1;
				for (int imageCol = 0; imageCol < image.cols; ++imageCol) {
					uchar bin1 = rowValues[imageCol][0];
					float weight1 = factor * rowValues[imageCol][1];
					uchar bin2 = rowValues[imageCol][2];
					float weight2 = factor * rowValues[imageCol][3];

					int colIndex0 = colCache[imageCol].index1;
					int colIndex1 = colCache[imageCol].index2;
					float colWeight1 = colCache[imageCol].weight2;
					float colWeight0 = colCache[imageCol].weight1;
					if (rowIndex0 >= 0 && colIndex0 >= 0) {
						float* histogramValues = histograms.ptr<float>(rowIndex0, colIndex0);
						histogramValues[bin1] += weight1 * rowWeight0 * colWeight0;
						histogramValues[bin2] += weight2 * rowWeight0 * colWeight0;
					}
					if (rowIndex0 >= 0 && colIndex1 < columnCount) {
						float* histogramValues = histograms.ptr<float>(rowIndex0, colIndex1);
						histogramValues[bin1] += weight1 * rowWeight0 * colWeight1;
						histogramValues[bin2] += weight2 * rowWeight0 * colWeight1;
					}
					if (rowIndex1 < rowCount && colIndex0 >= 0) {
						float* histogramValues = histograms.ptr<float>(rowIndex1, colIndex0);
						histogramValues[bin1] += weight1 * rowWeight1 * colWeight0;
						histogramValues[bin2] += weight2 * rowWeight1 * colWeight0;
					}
					if (rowIndex1 < rowCount && colIndex1 < columnCount) {
						float* histogramValues = histograms.ptr<float>(rowIndex1, colIndex1);
						histogramValues[bin1] += weight1 * rowWeight1 * colWeight1;
						histogramValues[bin2] += weight2 * rowWeight1 * colWeight1;
					}
				}
			}
		}
	} else { // no bilinear interpolation between cells
		float* histogramValues = histograms.ptr<float>();
		if (image.channels() == 1) { // bin information only, no weights
			for (int cellRow = 0; cellRow < rowCount; ++cellRow) {
				for (int cellCol = 0; cellCol < columnCount; ++cellCol) {
					int startRow = (cellRow * image.rows) / rowCount;
					int startCol = (cellCol * image.cols) / columnCount;
					int endRow = ((cellRow + 1) * image.rows) / rowCount;
					int endCol = ((cellCol + 1) * image.cols) / columnCount;
					for (int imageRow = startRow; imageRow < endRow; ++imageRow) {
						const uchar* rowValues = image.ptr<uchar>(imageRow);
						for (int imageCol = startCol; imageCol < endCol; ++imageCol)
							histogramValues[rowValues[imageCol]]++;
					}
					histogramValues += binCount;
				}
			}
		} else if (image.channels() == 2) { // bin index and weight available
			for (int cellRow = 0; cellRow < rowCount; ++cellRow) {
				for (int cellCol = 0; cellCol < columnCount; ++cellCol) {
					int startRow = (cellRow * image.rows) / rowCount;
					int startCol = (cellCol * image.cols) / columnCount;
					int endRow = ((cellRow + 1) * image.rows) / rowCount;
					int endCol = ((cellCol + 1) * image.cols) / columnCount;
					for (int imageRow = startRow; imageRow < endRow; ++imageRow) {
						const Vec2b* rowValues = image.ptr<Vec2b>(imageRow);
						for (int imageCol = startCol; imageCol < endCol; ++imageCol) {
							uchar bin = rowValues[imageCol][0];
							uchar weight = rowValues[imageCol][1];
							histogramValues[bin] += factor * weight;
						}
					}
					histogramValues += binCount;
				}
			}
		} else if (image.channels() == 4) { // two bin indices and weights available
			for (int cellRow = 0; cellRow < rowCount; ++cellRow) {
				for (int cellCol = 0; cellCol < columnCount; ++cellCol) {
					int startRow = (cellRow * image.rows) / rowCount;
					int startCol = (cellCol * image.cols) / columnCount;
					int endRow = ((cellRow + 1) * image.rows) / rowCount;
					int endCol = ((cellCol + 1) * image.cols) / columnCount;
					for (int imageRow = startRow; imageRow < endRow; ++imageRow) {
						const Vec4b* rowValues = image.ptr<Vec4b>(imageRow);
						for (int imageCol = startCol; imageCol < endCol; ++imageCol) {
							uchar bin1 = rowValues[imageCol][0];
							uchar weight1 = rowValues[imageCol][1];
							uchar bin2 = rowValues[imageCol][2];
							uchar weight2 = rowValues[imageCol][3];
							histogramValues[bin1] += factor * weight1;
							histogramValues[bin2] += factor * weight2;
						}
					}
					histogramValues += binCount;
				}
			}
		}
	}
}

void HistogramFilter::createCache(vector<CacheEntry>& cache, unsigned int size, int count) const {
	if (cache.size() != size) {
		cache.clear();
		cache.reserve(size);
		CacheEntry entry;
		for (unsigned int matIndex = 0; matIndex < size; ++matIndex) {
			double realIndex = static_cast<double>(count) * (static_cast<double>(matIndex) + 0.5) / static_cast<double>(size) - 0.5;
			entry.index1 = static_cast<int>(floor(realIndex));
			entry.index2 = entry.index1 + 1;
			entry.weight2 = realIndex - entry.index1;
			entry.weight1 = 1.f - entry.weight2;
			if (entry.index1 < 0) {
				entry.index1 = entry.index2;
				entry.weight1 = 0;
			} else if (entry.index2 >= static_cast<int>(count)) {
				entry.index2 = entry.index1;
				entry.weight2 = 0;
			}
			cache.push_back(entry);
		}
	}
}

void HistogramFilter::normalize(Mat& histogram) const {
	switch (normalization) {
		case Normalization::L2NORM: normalizeL2(histogram); break;
		case Normalization::L2HYS:  normalizeL2Hys(histogram); break;
		case Normalization::L1NORM: normalizeL1(histogram); break;
		case Normalization::L1SQRT: normalizeL1Sqrt(histogram); break;
		case Normalization::NONE:   break;
	}
}

void HistogramFilter::normalizeL2(Mat& histogram) const {
	float norm = cv::norm(histogram, cv::NORM_L2);
	histogram = histogram / (norm + eps);
}

void HistogramFilter::normalizeL2Hys(Mat& histogram) const {
	normalizeL2(histogram);
	histogram = min(histogram, 0.2);
	normalizeL2(histogram);
}

void HistogramFilter::normalizeL1(Mat& histogram) const {
	float norm = cv::norm(histogram, cv::NORM_L1);
	histogram = histogram / (norm + eps);
}

void HistogramFilter::normalizeL1Sqrt(Mat& histogram) const {
	normalizeL1(histogram);
	cv::sqrt(histogram, histogram);
}

} /* namespace imageprocessing */
