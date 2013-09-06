/*
 * HistogramFilter.cpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/HistogramFilter.hpp"
#include <stdexcept>

using cv::Vec2b;
using cv::Vec4b;
using std::runtime_error;

namespace imageprocessing {

const float HistogramFilter::eps = 1e-4;

HistogramFilter::HistogramFilter(Normalization normalization) : normalization(normalization),	rowCache(), colCache() {}

HistogramFilter::~HistogramFilter() {}

void HistogramFilter::createCellHistograms(const Mat& image, Mat& histograms, int binCount, int rowCount, int columnCount, bool interpolate) const {
	if (image.channels() != 1 && image.channels() != 2 && image.channels() != 4)
		throw runtime_error("HistogramFilter: image must have one, two or four channels");
	if (image.depth() != CV_8U)
		throw runtime_error("HistogramFilter: image must have a depth of CV_8U");

	histograms = Mat::zeros(1, rowCount * columnCount * binCount, CV_32F);
	float factor = 1.f / 255.f;
	if (interpolate) { // bilinear interpolation between cells
		float* histogramsValues = histograms.ptr<float>();
		createCache(rowCache, image.rows, rowCount);
		createCache(colCache, image.cols, columnCount);
		if (image.channels() == 1) { // bin information only, no weights
			for (int imageRow = 0; imageRow < image.rows; ++imageRow) {
				const uchar* rowValues = image.ptr<uchar>(imageRow);
				int rowIndex0 = rowCache[imageRow].index;
				int rowIndex1 = rowIndex0 + 1;
				float rowWeight0 = rowCache[imageRow].weight;
				float rowWeight1 = 1.f - rowWeight0;
				for (int imageCol = 0; imageCol < image.cols; ++imageCol) {
					uchar bin = rowValues[imageCol];

					int colIndex0 = colCache[imageCol].index;
					int colIndex1 = colIndex0 + 1;
					float colWeight0 = colCache[imageCol].weight;
					float colWeight1 = 1.f - colWeight0;
					if (colIndex0 >= 0 && rowIndex0 >= 0) {
						float* histogramValues = histogramsValues + rowIndex0 * columnCount * binCount + colIndex0 * binCount;
						histogramValues[bin] += colWeight1 * rowWeight1;
					}
					if (colIndex1 < columnCount && rowIndex0 >= 0) {
						float* histogramValues = histogramsValues + rowIndex0 * columnCount * binCount + colIndex1 * binCount;
						histogramValues[bin] += colWeight0 * rowWeight1;
					}
					if (colIndex0 >= 0 && rowIndex1 < rowCount) {
						float* histogramValues = histogramsValues + rowIndex1 * columnCount * binCount + colIndex0 * binCount;
						histogramValues[bin] += colWeight1 * rowWeight0;
					}
					if (colIndex1 < columnCount && rowIndex1 < rowCount) {
						float* histogramValues = histogramsValues + rowIndex1 * columnCount * binCount + colIndex1 * binCount;
						histogramValues[bin] += colWeight0 * rowWeight0;
					}
				}
			}
		} else if (image.channels() == 2) { // bin index and weight available
			for (int imageRow = 0; imageRow < image.rows; ++imageRow) {
				const Vec2b* rowValues = image.ptr<Vec2b>(imageRow);
				int rowInedx0 = rowCache[imageRow].index;
				int rowIndex1 = rowInedx0 + 1;
				float rowWeight0 = rowCache[imageRow].weight;
				float rowWeight1 = 1.f - rowWeight0;
				for (int imageCol = 0; imageCol < image.cols; ++imageCol) {
					uchar bin = rowValues[imageCol][0];
					float weight = factor * rowValues[imageCol][1];

					int colIndex0 = colCache[imageCol].index;
					int colIndex1 = colIndex0 + 1;
					float colWeight0 = colCache[imageCol].weight;
					float colWeight1 = 1.f - colWeight0;
					if (colIndex0 >= 0 && rowInedx0 >= 0) {
						float* histogramValues = histogramsValues + rowInedx0 * columnCount * binCount + colIndex0 * binCount;
						histogramValues[bin] += weight * colWeight1 * rowWeight1;
					}
					if (colIndex1 < columnCount && rowInedx0 >= 0) {
						float* histogramValues = histogramsValues + rowInedx0 * columnCount * binCount + colIndex1 * binCount;
						histogramValues[bin] += weight * colWeight0 * rowWeight1;
					}
					if (colIndex0 >= 0 && rowIndex1 < rowCount) {
						float* histogramValues = histogramsValues + rowIndex1 * columnCount * binCount + colIndex0 * binCount;
						histogramValues[bin] += weight * colWeight1 * rowWeight0;
					}
					if (colIndex1 < columnCount && rowIndex1 < rowCount) {
						float* histogramValues = histogramsValues + rowIndex1 * columnCount * binCount + colIndex1 * binCount;
						histogramValues[bin] += weight * colWeight0 * rowWeight0;
					}
				}
			}
		} else if (image.channels() == 4) { // two bin indices and weights available
			for (int imageRow = 0; imageRow < image.rows; ++imageRow) {
				const Vec4b* rowValues = image.ptr<Vec4b>(imageRow);
				int rowIndex0 = rowCache[imageRow].index;
				int rowIndex1 = rowIndex0 + 1;
				float rowWeight0 = rowCache[imageRow].weight;
				float rowWeight1 = 1.f - rowWeight0;
				for (int imageCol = 0; imageCol < image.cols; ++imageCol) {
					uchar bin1 = rowValues[imageCol][0];
					float weight1 = factor * rowValues[imageCol][1];
					uchar bin2 = rowValues[imageCol][2];
					float weight2 = factor * rowValues[imageCol][3];

					int colIndex0 = colCache[imageCol].index;
					int colIndex1 = colIndex0 + 1;
					float colWeight0 = colCache[imageCol].weight;
					float colWeight1 = 1.f - colWeight0;
					if (colIndex0 >= 0 && rowIndex0 >= 0) {
						float* histogramValues = histogramsValues + rowIndex0 * columnCount * binCount + colIndex0 * binCount;
						histogramValues[bin1] += weight1 * colWeight1 * rowWeight1;
						histogramValues[bin2] += weight2 * colWeight1 * rowWeight1;
					}
					if (colIndex1 < columnCount && rowIndex0 >= 0) {
						float* histogramValues = histogramsValues + rowIndex0 * columnCount * binCount + colIndex1 * binCount;
						histogramValues[bin1] += weight1 * colWeight0 * rowWeight1;
						histogramValues[bin2] += weight2 * colWeight0 * rowWeight1;
					}
					if (colIndex0 >= 0 && rowIndex1 < rowCount) {
						float* histogramValues = histogramsValues + rowIndex1 * columnCount * binCount + colIndex0 * binCount;
						histogramValues[bin1] += weight1 * colWeight1 * rowWeight0;
						histogramValues[bin2] += weight2 * colWeight1 * rowWeight0;
					}
					if (colIndex1 < columnCount && rowIndex1 < rowCount) {
						float* histogramValues = histogramsValues + rowIndex1 * columnCount * binCount + colIndex1 * binCount;
						histogramValues[bin1] += weight1 * colWeight0 * rowWeight0;
						histogramValues[bin2] += weight2 * colWeight0 * rowWeight0;
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
			entry.index = static_cast<int>(floor(realIndex));
			entry.weight = realIndex - entry.index;
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
