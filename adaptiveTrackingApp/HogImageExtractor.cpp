/*
 * DirectImageExtractor.cpp
 *
 *  Created on: 30.06.2013
 *      Author: ex-ratt
 */

#include "HogImageExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream> // TODO
using cv::Rect;
using cv::Size;
using cv::resize;

namespace imageprocessing {

HogImageExtractor::HogImageExtractor(unsigned int bins, double offset, bool signedGradients, int cellCount, int blockSize, bool combineHistograms, Normalization normalization) :
		HistogramFeatureExtractor(normalization),
		hogImage(), version(-1), image(), filter(), gradientFilter(1, 0), bins(bins), offset(offset),
		cellCols(cellCount),
		cellRows(cellCount),
		blockWidth(blockSize),
		blockHeight(blockSize),
		combineHistograms(combineHistograms) {
	integralHistogram.reserve(bins);
	for (unsigned int i = 0; i < bins; ++i)
		integralHistogram.push_back(Mat());
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	BinEntry binEntry;
	// build the look-up table
	// index of the look-up table is the binary concatanation of the gradients of x and y
	// value of the look-up table is the binary concatanation of the bin index and weight (scaled to 255)
	gradientCode.gradient.x = 0;
	for (int x = 0; x < 256; ++x) {
		double gradientX = (static_cast<double>(x) - 127) / 127;
		gradientCode.gradient.y = 0;
		for (int y = 0; y < 256; ++y) {
			double gradientY = (static_cast<double>(y) - 127) / 127;
			double direction = atan2(gradientY, gradientX);
			double magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);
			double bin;
			if (signedGradients) {
				direction += CV_PI;
				bin = (direction + offset) * bins / (2 * CV_PI);
			} else { // unsigned gradients
				if (direction < 0)
					direction += CV_PI;
				bin = (direction + offset) * bins / CV_PI;
			}
			binEntry.bin1 = static_cast<int>(floor(bin)) % bins;
			binEntry.bin2 = static_cast<int>(ceil(bin)) % bins;
			binEntry.weight1 = (bin - binEntry.bin1) * magnitude;
			if (binEntry.bin1 == binEntry.bin2)
				binEntry.weight2 = 0;
			else
				binEntry.weight2 = abs(binEntry.bin2 - bin) * magnitude;
			binData[gradientCode.index] = binEntry;
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

HogImageExtractor::~HogImageExtractor() {}

void HogImageExtractor::update(const Mat& image) {
	buildHistogram(filter.applyTo(image));
	version = -1;
}

void HogImageExtractor::update(shared_ptr<VersionedImage> image) {
	if (version != image->getVersion()) {
		buildHistogram(filter.applyTo(image->getData()));
		version = image->getVersion();
	}
}

void HogImageExtractor::buildHistogram(const Mat& image) {
	this->image = image;
	Mat gradientImage = gradientFilter.applyTo(image);
	int rows = gradientImage.rows;
	int cols = gradientImage.cols;
	Mat histogramBin(rows, cols, CV_32F);
	if (image.isContinuous() && histogramBin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	for (unsigned int bin = 0; bin < bins; ++bin) {
		histogramBin.setTo(0);
		for (int row = 0; row < rows; ++row) {
			const ushort* gradientCode = gradientImage.ptr<ushort>(row); // concatenation of x gradient and y gradient (both uchar)
			float* histogramBinRow = histogramBin.ptr<float>(row);
			for (int col = 0; col < cols; ++col) {
				BinEntry& binCode = binData[gradientCode[col]];
				if (binCode.bin1 == bin)
					histogramBinRow[col] = binCode.weight1;
				else if (binCode.bin2 == bin)
					histogramBinRow[col] = binCode.weight2;
			}
		}
		integralHistogram[bin].create(gradientImage.rows + 1, gradientImage.cols + 1, CV_32F);
		integral(histogramBin, integralHistogram[bin], CV_32F);
	}
}

shared_ptr<Patch> HogImageExtractor::extract(int x, int y, int width, int height) const {
	int px = x - width / 2;
	int py = y - height / 2;
	if (px < 0 || py < 0 || px + width >= image.cols || py + height >= image.rows)
		return shared_ptr<Patch>();

	// create histograms of cells
	vector<Mat> cellHistograms(cellRows * cellCols);
	for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
		for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
			Mat& histogram = cellHistograms[cellRow * cellCols + cellCol];
			histogram = Mat::zeros(1, bins, CV_32F);
			float* histogramValues = histogram.ptr<float>(0);
			int startRow = py + (cellRow * height) / cellRows;
			int startCol = px + (cellCol * width) / cellCols;
			int endRow = py + ((cellRow + 1) * height) / cellRows;
			int endCol = px + ((cellCol + 1) * width) / cellCols;
			for (int bin = 0; bin < integralHistogram.size(); ++bin) {
				const Mat& binData = integralHistogram[bin];
				histogramValues[bin] =
						binData.at<float>(startRow, startCol)
						+ binData.at<float>(endRow, endCol)
						- binData.at<float>(startRow, endCol)
						- binData.at<float>(endRow, startCol);
			}
		}
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

			normalize(blockHistogram);
		}
	}
	Mat patchData(1, blockHistograms.size() * blockHistogramSize, CV_32F);
	float* patchValues = patchData.ptr<float>(0);
	for (unsigned int histIndex = 0; histIndex < blockHistograms.size(); ++histIndex) {
		Mat& blockHistogram = blockHistograms[histIndex];
		float* blockHistogramValues = blockHistogram.ptr<float>(0);
		for (int bin = 0; bin < blockHistogramSize; ++bin)
			patchValues[bin] = blockHistogramValues[bin];
		patchValues += blockHistogramSize;
	}

//	hogImage = Mat::zeros(20 * cellRows, 20 * cellCols, CV_8U);
//	for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
//		for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
//			Mat& histogram = cellHistograms[cellRow * cellCols + cellCol];
//			for (int i = 0; i < histogram.cols; ++i) {
//				double angle = (i + 0.5) * CV_PI / bins - offset;
//				double x = cos(angle);
//				double y = sin(angle);
//				double xStart = 10 * x + 10 + 20 * cellCol;
//				double xEnd = -10 * x + 10 + 20 * cellCol;
//				double yStart = 10 * y + 10 + 20 * cellRow;
//				double yEnd = -10 * y + 10 + 20 * cellRow;
//				float weight = histogram.at<float>(i) / 20;
//				cv::Scalar color(255 * weight, 255 * weight, 255 * weight);
//				cv::line(hogImage, cv::Point(xStart, yStart), cv::Point(xEnd, yEnd), color);
//			}
//		}
//	}

	return make_shared<Patch>(x, y, width, height, patchData);
}

} /* namespace imageprocessing */
