/*
 * ExtendedHogFilter.cpp
 *
 *  Created on: 01.08.2013
 *      Author: poschmann
 */

#include "imageprocessing/ExtendedHogFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::invalid_argument;

namespace imageprocessing {

ExtendedHogFilter::ExtendedHogFilter(int binCount, int cellSize, bool interpolate, bool signedAndUnsigned, float alpha) :
		HistogramFilter(Normalization::L2HYS),
		binCount(binCount),
		cellWidth(cellSize),
		cellHeight(cellSize),
		interpolate(interpolate),
		signedAndUnsigned(signedAndUnsigned),
		alpha(alpha) {
	if (binCount <= 0)
		throw invalid_argument("ExtendedHogFilter: binCount must be greater than zero");
	if (cellSize <= 0)
		throw invalid_argument("ExtendedHogFilter: cellSize must be greater than zero");
	if (signedAndUnsigned && binCount % 2 != 0)
		throw invalid_argument("ExtendedHogFilter: the bin size must be even for signed and unsigned gradients to be combined");
	if (alpha <= 0)
		throw invalid_argument("ExtendedHogFilter: alpha must be greater than zero");
}

ExtendedHogFilter::ExtendedHogFilter(int binCount, int cellWidth, int cellHeight, bool interpolate, bool signedAndUnsigned, float alpha) :
		HistogramFilter(Normalization::L2HYS),
		binCount(binCount),
		cellWidth(cellWidth),
		cellHeight(cellHeight),
		interpolate(interpolate),
		signedAndUnsigned(signedAndUnsigned),
		alpha(alpha) {
	if (binCount <= 0)
		throw invalid_argument("ExtendedHogFilter: binCount must be greater than zero");
	if (cellWidth <= 0)
		throw invalid_argument("ExtendedHogFilter: cellWidth must be greater than zero");
	if (cellHeight <= 0)
		throw invalid_argument("ExtendedHogFilter: cellHeight must be greater than zero");
	if (signedAndUnsigned && binCount % 2 != 0)
		throw invalid_argument("ExtendedHogFilter: the bin size must be even for signed and unsigned gradients to be combined");
	if (alpha <= 0)
		throw invalid_argument("ExtendedHogFilter: alpha must be greater than zero");
}

Mat ExtendedHogFilter::applyTo(const Mat& image, Mat& filtered) const {
	int cellRowCount = cvRound(static_cast<double>(image.rows) / static_cast<double>(cellHeight));
	int cellColumnCount = cvRound(static_cast<double>(image.cols) / static_cast<double>(cellWidth));
	Mat cellHistograms;
	createCellHistograms(image, cellHistograms, binCount, cellRowCount, cellColumnCount, interpolate);
	createDescriptors(cellHistograms, filtered, binCount, signedAndUnsigned, cellRowCount, cellColumnCount, alpha);
	return filtered;
}

void ExtendedHogFilter::createDescriptors(const Mat& histograms, Mat& descriptors,
		int binCount, bool signedAndUnsigned, int cellRowCount, int cellColumnCount, float alpha) const {
	Mat energies = Mat::zeros(cellRowCount, cellColumnCount, CV_32F);
	const float* histogramsValues = histograms.ptr<float>();
	float* energiesValues = energies.ptr<float>();

	// create extended HOG feature vector
	if (signedAndUnsigned) { // signed and unsigned gradients should be combined into descriptor

		// compute gradient energy over cells
		int binHalfCount = binCount / 2;
		for (int cellIndex = 0; cellIndex < cellRowCount * cellColumnCount; ++cellIndex) {
			const float* histogramValues = histogramsValues + cellIndex * binCount;
			for (int binIndex = 0; binIndex < binHalfCount; ++binIndex) {
				float sum = histogramValues[binIndex] + histogramValues[binIndex + binHalfCount];
				energiesValues[cellIndex] += sum * sum;
			}
		}

		// create descriptors
		descriptors.create(cellRowCount, cellColumnCount, CV_32FC(binCount + binHalfCount + 4));
		float* values = descriptors.ptr<float>();
		for (int cellRow = 0; cellRow < cellRowCount; ++cellRow) {
			for (int cellCol = 0; cellCol < cellColumnCount; ++cellCol) {
				const float* cellHistogramValues = histogramsValues + cellRow * cellColumnCount * binCount + cellCol * binCount;
				int r1 = cellRow;
				int r0 = std::max(0, cellRow - 1);
				int r2 = std::min(cellRow + 1, cellRowCount - 1);
				int c1 = cellCol;
				int c0 = std::max(0, cellCol - 1);
				int c2 = std::min(cellCol + 1, cellColumnCount - 1);
				float sqn00 = energies.at<float>(r0, c0);
				float sqn01 = energies.at<float>(r0, c1);
				float sqn02 = energies.at<float>(r0, c2);
				float sqn10 = energies.at<float>(r1, c0);
				float sqn11 = energies.at<float>(r1, c1);
				float sqn12 = energies.at<float>(r1, c2);
				float sqn20 = energies.at<float>(r2, c0);
				float sqn21 = energies.at<float>(r2, c1);
				float sqn22 = energies.at<float>(r2, c2);
				float n1 = 1.f / sqrt(sqn00 + sqn01 + sqn10 + sqn11 + eps);
				float n2 = 1.f / sqrt(sqn01 + sqn02 + sqn11 + sqn12 + eps);
				float n3 = 1.f / sqrt(sqn10 + sqn11 + sqn20 + sqn21 + eps);
				float n4 = 1.f / sqrt(sqn11 + sqn12 + sqn21 + sqn22 + eps);

				float t1 = 0;
				float t2 = 0;
				float t3 = 0;
				float t4 = 0;

				// signed orientation features (aka contrast-sensitive)
				for (int binIndex = 0; binIndex < binCount; ++binIndex) {
					float h1 = std::min(alpha, cellHistogramValues[binIndex] * n1);
					float h2 = std::min(alpha, cellHistogramValues[binIndex] * n2);
					float h3 = std::min(alpha, cellHistogramValues[binIndex] * n3);
					float h4 = std::min(alpha, cellHistogramValues[binIndex] * n4);
					values[binIndex] = 0.5 * (h1 + h2 + h3 + h4);
					t1 += h1;
					t2 += h2;
					t3 += h3;
					t4 += h4;
				}
				values += binCount;

				// unsigned orientation features (aka contrast-insensitive)
				for (int binIndex = 0; binIndex < binHalfCount; ++binIndex) {
					float sum = cellHistogramValues[binIndex] + cellHistogramValues[binIndex + binHalfCount];
					float h1 = std::min(alpha, sum * n1);
					float h2 = std::min(alpha, sum * n2);
					float h3 = std::min(alpha, sum * n3);
					float h4 = std::min(alpha, sum * n4);
					values[binIndex] = 0.5 * (h1 + h2 + h3 + h4);
				}
				values += binHalfCount;

				// energy features
				values[0] = 0.2357 * t1;
				values[1] = 0.2357 * t2;
				values[2] = 0.2357 * t3;
				values[3] = 0.2357 * t4;
				values += 4;
			}
		}
	} else { // only signed or unsigned gradients should be in descriptor

		// compute gradient energy over cells
		for (int cellIndex = 0; cellIndex < cellRowCount * cellColumnCount; ++cellIndex) {
			const float* histogramValues = histogramsValues + cellIndex * binCount;
			for (int binIndex = 0; binIndex < binCount; ++binIndex)
				energiesValues[cellIndex] += histogramValues[binIndex] * histogramValues[binIndex];
		}

		// create descriptors
		descriptors.create(cellRowCount, cellColumnCount, CV_32FC(binCount + 4));
		float* values = descriptors.ptr<float>();
		for (int cellRow = 0; cellRow < cellRowCount; ++cellRow) {
			for (int cellCol = 0; cellCol < cellColumnCount; ++cellCol) {
				const float* cellHistogramValues = histogramsValues + cellRow * cellColumnCount * binCount + cellCol * binCount;
				int r1 = cellRow;
				int r0 = std::max(cellRow - 1, 0);
				int r2 = std::min(cellRow + 1, cellRowCount - 1);
				int c1 = cellCol;
				int c0 = std::max(cellCol - 1, 0);
				int c2 = std::min(cellCol + 1, cellColumnCount - 1);
				float sqn00 = energies.at<float>(r0, c0);
				float sqn01 = energies.at<float>(r0, c1);
				float sqn02 = energies.at<float>(r0, c2);
				float sqn10 = energies.at<float>(r1, c0);
				float sqn11 = energies.at<float>(r1, c1);
				float sqn12 = energies.at<float>(r1, c2);
				float sqn20 = energies.at<float>(r2, c0);
				float sqn21 = energies.at<float>(r2, c1);
				float sqn22 = energies.at<float>(r2, c2);
				float n1 = 1.f / sqrt(sqn00 + sqn01 + sqn10 + sqn11 + eps);
				float n2 = 1.f / sqrt(sqn01 + sqn02 + sqn11 + sqn12 + eps);
				float n3 = 1.f / sqrt(sqn10 + sqn11 + sqn20 + sqn21 + eps);
				float n4 = 1.f / sqrt(sqn11 + sqn12 + sqn21 + sqn22 + eps);

				float t1 = 0;
				float t2 = 0;
				float t3 = 0;
				float t4 = 0;

				// orientation features
				for (int binIndex = 0; binIndex < binCount; ++binIndex) {
					float h1 = std::min(alpha, cellHistogramValues[binIndex] * n1);
					float h2 = std::min(alpha, cellHistogramValues[binIndex] * n2);
					float h3 = std::min(alpha, cellHistogramValues[binIndex] * n3);
					float h4 = std::min(alpha, cellHistogramValues[binIndex] * n4);
					values[binIndex] = 0.5 * (h1 + h2 + h3 + h4);
					t1 += h1;
					t2 += h2;
					t3 += h3;
					t4 += h4;
				}

				// energy features
				values += binCount;
				values[0] = 0.2357 * t1;
				values[1] = 0.2357 * t2;
				values[2] = 0.2357 * t3;
				values[3] = 0.2357 * t4;
				values += 4;
			}
		}
	}
}

} /* namespace imageprocessing */
