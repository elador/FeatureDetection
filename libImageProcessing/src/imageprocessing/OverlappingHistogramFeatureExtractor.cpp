/*
 * OverlappingHistogramFeatureExtractor.cpp
 *
 *  Created on: 30.05.2013
 *      Author: poschmann
 */

#include "imageprocessing/OverlappingHistogramFeatureExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include <vector>
#include <stdexcept>

using cv::Vec2b;
using std::vector;
using std::runtime_error;

namespace imageprocessing {

OverlappingHistogramFeatureExtractor::OverlappingHistogramFeatureExtractor(shared_ptr<FeatureExtractor> extractor,
		unsigned int bins, int cellSize, int blockSize, Normalization normalization) :
				extractor(extractor),
				bins(bins),
				cellWidth(cellSize),
				cellHeight(cellSize),
				blockWidth(blockSize),
				blockHeight(blockSize),
				normalization(normalization) {}

OverlappingHistogramFeatureExtractor::OverlappingHistogramFeatureExtractor(
		shared_ptr<FeatureExtractor> extractor, unsigned int bins,
		int cellWidth, int cellHeight, int blockWidth, int blockHeight, Normalization normalization) :
				extractor(extractor),
				bins(bins),
				cellWidth(cellWidth),
				cellHeight(cellHeight),
				blockWidth(blockWidth),
				blockHeight(blockHeight),
				normalization(normalization) {}

OverlappingHistogramFeatureExtractor::~OverlappingHistogramFeatureExtractor() {}

void OverlappingHistogramFeatureExtractor::update(const Mat& image) {
	extractor->update(image);
}

void OverlappingHistogramFeatureExtractor::update(shared_ptr<VersionedImage> image) {
	extractor->update(image);
}

shared_ptr<Patch> OverlappingHistogramFeatureExtractor::extract(int x, int y, int width, int height) const {
	shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
	if (patch) {
		bool combineHistograms = false;
		Mat& patchData = patch->getData();

		// create histograms of cells
		int cellRows = cvRound(static_cast<double>(patchData.rows) / static_cast<double>(cellHeight));
		int cellCols = cvRound(static_cast<double>(patchData.cols) / static_cast<double>(cellWidth));
		vector<Mat> cellHistograms(cellRows * cellCols);
		if (patchData.channels() == 1) { // bin information only, no weights
			for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
				for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
					Mat& histogram = cellHistograms[cellRow * cellCols + cellCol];
					histogram = Mat::zeros(1, bins, CV_32F);
					float* histogramValues = histogram.ptr<float>(0);
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
					Mat& histogram = cellHistograms[cellRow * cellCols + cellCol];
					histogram = Mat::zeros(1, bins, CV_32F);
					float* histogramValues = histogram.ptr<float>(0);
					int startRow = (cellRow * patchData.rows) / cellRows;
					int startCol = (cellCol * patchData.cols) / cellCols;
					int endRow = ((cellRow + 1) * patchData.rows) / cellRows;
					int endCol = ((cellCol + 1) * patchData.cols) / cellCols;
					for (int row = startRow; row < endRow; ++row) {
						Vec2b* rowValues = patchData.ptr<Vec2b>(row);
						for (int col = startCol; col < endCol; ++col) {
							uchar bin = rowValues[col][0];
							uchar weight = rowValues[col][1];
							histogramValues[bin] += weight / 255.0f;
						}
					}
				}
			}
		} else {
			throw runtime_error("OverlappingHistogramFeatureExtractor: Patch data must have one or two channels");
		}

		// create histograms of blocks
		int blockHistogramSize;
		int blockRows = cellRows - blockHeight + 1;
		int blockCols = cellCols - blockWidth + 1;
		vector<Mat> blockHistograms(blockRows * blockCols);
		for (int blockRow = 0; blockRow < blockRows; ++blockRow) {
			for (int blockCol = 0; blockCol < blockCols; ++blockCol) {
				Mat& blockHistogram = blockHistograms[blockRow * blockCols + blockCol];
				if (combineHistograms) { // combine histograms by adding the bin values
					blockHistogramSize = bins;
					blockHistogram = Mat::zeros(1, bins, CV_32F);
					float* blockHistogramValues = blockHistogram.ptr<float>(0);
					for (int cellRow = blockRow; cellRow < blockRow + blockHeight; ++cellRow) {
						for (int cellCol = blockCol; cellCol < blockCol + blockWidth; ++cellCol) {
							Mat& cellHistogram = cellHistograms[blockRow * blockCols + blockCol];
							float* cellHistogramValues = cellHistogram.ptr<float>(0);
							for (unsigned int bin = 0; bin < bins; ++bin)
								blockHistogramValues[bin] += cellHistogramValues[bin];
						}
					}
				} else { // create concatenation of histograms
					blockHistogramSize = blockWidth * blockHeight * bins;
					blockHistogram.create(1, blockHistogramSize, CV_32F);
					float* blockHistogramValues = blockHistogram.ptr<float>(0);
					for (int cellRow = blockRow; cellRow < blockRow + blockHeight; ++cellRow) {
						for (int cellCol = blockCol; cellCol < blockCol + blockWidth; ++cellCol) {
							Mat& cellHistogram = cellHistograms[blockRow * blockCols + blockCol];
							float* cellHistogramValues = cellHistogram.ptr<float>(0);
							for (unsigned int bin = 0; bin < bins; ++bin)
								blockHistogramValues[bin] = cellHistogramValues[bin];
							blockHistogramValues += bins;
						}
					}
				}

				switch (normalization) {
				case L2NORM: normalizeL2(blockHistogram); break;
				case L2HYS:  normalizeL2Hys(blockHistogram); break;
				case L1NORM: normalizeL1(blockHistogram); break;
				case L1SQRT: normalizeL1Sqrt(blockHistogram); break;
				case NONE:   break;
				}
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

const float OverlappingHistogramFeatureExtractor::eps = 1e-3;

void OverlappingHistogramFeatureExtractor::normalizeL2(Mat& histogram) const {
	float normSquared = cv::norm(histogram, cv::NORM_L2SQR);
	histogram = histogram / sqrt(normSquared + eps * eps);
}

void OverlappingHistogramFeatureExtractor::normalizeL2Hys(Mat& histogram) const {
	normalizeL2(histogram);
	histogram = min(histogram, 0.2);
	normalizeL2(histogram);
}

void OverlappingHistogramFeatureExtractor::normalizeL1(Mat& histogram) const {
	float norm = cv::norm(histogram, cv::NORM_L1);
	histogram = histogram / (norm + eps);
}

void OverlappingHistogramFeatureExtractor::normalizeL1Sqrt(Mat& histogram) const {
	normalizeL1(histogram);
	cv::sqrt(histogram, histogram);
}

} /* namespace imageprocessing */
