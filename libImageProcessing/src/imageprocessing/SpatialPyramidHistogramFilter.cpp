/*
 * SpatialPyramidHistogramFilter.cpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/SpatialPyramidHistogramFilter.hpp"
#include <vector>
#include <stdexcept>

using cv::Vec2b;
using cv::Vec4b;
using std::vector;
using std::runtime_error;

namespace imageprocessing {

SpatialPyramidHistogramFilter::SpatialPyramidHistogramFilter(unsigned int bins, unsigned int level, Normalization normalization) :
		HistogramFilter(normalization),
		bins(bins),
		level(level) {}

SpatialPyramidHistogramFilter::~SpatialPyramidHistogramFilter() {}

Mat SpatialPyramidHistogramFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.depth() != CV_8U)
		throw runtime_error("SpatialPyramidHistogramFilter: image must have a depth of CV_8U");

	int count = 1 << (level - 1);
	vector<Mat> cellHistograms(count * count);
	float factor = 1.f / 255.f;
	if (image.channels() == 1) { // bin information only, no weights
		for (int cellRow = 0; cellRow < count; ++cellRow) {
			for (int cellCol = 0; cellCol < count; ++cellCol) {
				Mat& histogram = cellHistograms[cellRow * count + cellCol];
				histogram = Mat::zeros(1, bins, CV_32F);
				float* histogramValues = histogram.ptr<float>(0);
				int startRow = (cellRow * image.rows) / count;
				int startCol = (cellCol * image.cols) / count;
				int endRow = ((cellRow + 1) * image.rows) / count;
				int endCol = ((cellCol + 1) * image.cols) / count;
				for (int row = startRow; row < endRow; ++row) {
					const uchar* rowValues = image.ptr<uchar>(row);
					for (int col = startCol; col < endCol; ++col)
						histogramValues[rowValues[col]]++;
				}
				normalize(histogram);
			}
		}
	} else if (image.channels() == 2) { // bin index and weight available
		for (int cellRow = 0; cellRow < count; ++cellRow) {
			for (int cellCol = 0; cellCol < count; ++cellCol) {
				Mat& histogram = cellHistograms[cellRow * count + cellCol];
				histogram = Mat::zeros(1, bins, CV_32F);
				float* histogramValues = histogram.ptr<float>(0);
				int startRow = (cellRow * image.rows) / count;
				int startCol = (cellCol * image.cols) / count;
				int endRow = ((cellRow + 1) * image.rows) / count;
				int endCol = ((cellCol + 1) * image.cols) / count;
				for (int row = startRow; row < endRow; ++row) {
					const Vec2b* rowValues = image.ptr<Vec2b>(row);
					for (int col = startCol; col < endCol; ++col) {
						uchar bin = rowValues[col][0];
						uchar weight = rowValues[col][1];
						histogramValues[bin] += factor * weight;
					}
				}
				normalize(histogram);
			}
		}
	} else if (image.channels() == 4) { // two bin indices and weights available
		for (int cellRow = 0; cellRow < count; ++cellRow) {
			for (int cellCol = 0; cellCol < count; ++cellCol) {
				Mat& histogram = cellHistograms[cellRow * count + cellCol];
				histogram = Mat::zeros(1, bins, CV_32F);
				float* histogramValues = histogram.ptr<float>(0);
				int startRow = (cellRow * image.rows) / count;
				int startCol = (cellCol * image.cols) / count;
				int endRow = ((cellRow + 1) * image.rows) / count;
				int endCol = ((cellCol + 1) * image.cols) / count;
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
				normalize(histogram);
			}
		}
	} else {
		throw runtime_error("SpatialPyramidHistogramFilter: image must have one, two or four channels");
	}

	int histogramCount = 0;
	for (unsigned int l = 0; l < level; ++l)
		histogramCount += 1 << (2 * l);
	filtered.create(1, histogramCount * bins, CV_32F);
	float* histogramValues = filtered.ptr<float>() + histogramCount * bins;
	int cellCount = count;
	for (unsigned int l = level - 1; l > 0; --l) {
		histogramValues -= (1 << (2 * l)) * bins;
		int blockCount = cellCount / 2;
		vector<Mat> blockHistograms(blockCount * blockCount);
		for (int blockRow = 0; blockRow < blockCount; ++blockRow) {
			for (int blockCol = 0; blockCol < blockCount; ++blockCol) {
				Mat& blockHistogram = blockHistograms[blockRow * blockCount + blockCol];
				blockHistogram = Mat::zeros(1, bins, CV_32F);
				float* blockHistogramValues = blockHistogram.ptr<float>(0);
				for (int cellRow = 2 * blockRow; cellRow < 2 * (blockRow + 1); ++cellRow) {
					for (int cellCol = 2 * blockCol; cellCol < 2 * (blockCol + 1); ++cellCol) {
						int cellHistogramIndex = cellRow * cellCount + cellCol;
						const Mat& cellHistogram = cellHistograms[cellHistogramIndex];
						const float* cellHistogramValues = cellHistogram.ptr<float>(0);
						for (unsigned int b = 0; b < bins; ++b) {
							histogramValues[cellHistogramIndex * bins + b] = cellHistogramValues[b];
							blockHistogramValues[b] += cellHistogramValues[b];
						}
					}
				}
				normalize(blockHistogram);
			}
		}
		cellCount = blockCount;
		cellHistograms = std::move(blockHistograms);
	}

	// fill level 0
	const Mat& cellHistogram = cellHistograms[0];
	const float* cellHistogramValues = cellHistogram.ptr<float>();
	histogramValues -= bins;
	for (unsigned int b = 0; b < bins; ++b)
		histogramValues[b] = cellHistogramValues[b];

	return filtered;
}

} /* namespace imageprocessing */
