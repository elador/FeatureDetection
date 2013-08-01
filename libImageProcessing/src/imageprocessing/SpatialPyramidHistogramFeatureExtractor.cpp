/*
 * SpatialPyramidHistogramFeatureExtractor.cpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/SpatialPyramidHistogramFeatureExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include <vector>
#include <stdexcept>

using cv::Vec2b;
using cv::Vec4b;
using std::vector;
using std::runtime_error;

namespace imageprocessing {

SpatialPyramidHistogramFeatureExtractor::SpatialPyramidHistogramFeatureExtractor(shared_ptr<FeatureExtractor> extractor,
		unsigned int bins, unsigned int level, Normalization normalization) :
				HistogramFeatureExtractor(normalization),
				extractor(extractor),
				bins(bins),
				level(level) {}

SpatialPyramidHistogramFeatureExtractor::~SpatialPyramidHistogramFeatureExtractor() {}

void SpatialPyramidHistogramFeatureExtractor::update(const Mat& image) {
	extractor->update(image);
}

void SpatialPyramidHistogramFeatureExtractor::update(shared_ptr<VersionedImage> image) {
	extractor->update(image);
}

shared_ptr<Patch> SpatialPyramidHistogramFeatureExtractor::extract(int x, int y, int width, int height) const {
	shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
	if (patch) {
		Mat& patchData = patch->getData();
		int count = 1 << (level - 1);
		vector<Mat> cellHistograms(count * count);
		float factor = 1.f / 255.f;
		if (patchData.channels() == 1) { // bin information only, no weights
			for (int cellRow = 0; cellRow < count; ++cellRow) {
				for (int cellCol = 0; cellCol < count; ++cellCol) {
					Mat& histogram = cellHistograms[cellRow * count + cellCol];
					histogram = Mat::zeros(1, bins, CV_32F);
					float* histogramValues = histogram.ptr<float>(0);
					int startRow = (cellRow * patchData.rows) / count;
					int startCol = (cellCol * patchData.cols) / count;
					int endRow = ((cellRow + 1) * patchData.rows) / count;
					int endCol = ((cellCol + 1) * patchData.cols) / count;
					for (int row = startRow; row < endRow; ++row) {
						uchar* rowValues = patchData.ptr<uchar>(row);
						for (int col = startCol; col < endCol; ++col)
							histogramValues[rowValues[col]]++;
					}
					normalize(histogram);
				}
			}
		} else if (patchData.channels() == 2) { // bin index and weight available
			for (int cellRow = 0; cellRow < count; ++cellRow) {
				for (int cellCol = 0; cellCol < count; ++cellCol) {
					Mat& histogram = cellHistograms[cellRow * count + cellCol];
					histogram = Mat::zeros(1, bins, CV_32F);
					float* histogramValues = histogram.ptr<float>(0);
					int startRow = (cellRow * patchData.rows) / count;
					int startCol = (cellCol * patchData.cols) / count;
					int endRow = ((cellRow + 1) * patchData.rows) / count;
					int endCol = ((cellCol + 1) * patchData.cols) / count;
					for (int row = startRow; row < endRow; ++row) {
						Vec2b* rowValues = patchData.ptr<Vec2b>(row);
						for (int col = startCol; col < endCol; ++col) {
							uchar bin = rowValues[col][0];
							uchar weight = rowValues[col][1];
							histogramValues[bin] += factor * weight;
						}
					}
					normalize(histogram);
				}
			}
		} else if (patchData.channels() == 4) { // two bin indices and weights available
			for (int cellRow = 0; cellRow < count; ++cellRow) {
				for (int cellCol = 0; cellCol < count; ++cellCol) {
					Mat& histogram = cellHistograms[cellRow * count + cellCol];
					histogram = Mat::zeros(1, bins, CV_32F);
					float* histogramValues = histogram.ptr<float>(0);
					int startRow = (cellRow * patchData.rows) / count;
					int startCol = (cellCol * patchData.cols) / count;
					int endRow = ((cellRow + 1) * patchData.rows) / count;
					int endCol = ((cellCol + 1) * patchData.cols) / count;
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
					normalize(histogram);
				}
			}
		} else {
			throw runtime_error("SpatialPyramidHistogramFeatureExtractor: Patch data must have one, two or four channels");
		}

		int histogramCount = 0;
		for (unsigned int l = 0; l < level; ++l)
			histogramCount += 1 << (2 * l);
		patch->getData().create(1, histogramCount * bins, CV_32F);
		Mat& histogram = patch->getData();
		float* histogramValues = histogram.ptr<float>(0) + histogramCount * bins;
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
		const float* cellHistogramValues = cellHistogram.ptr<float>(0);
		histogramValues -= bins;
		for (unsigned int b = 0; b < bins; ++b)
			histogramValues[b] = cellHistogramValues[b];
	}
	return patch;
}

} /* namespace imageprocessing */
