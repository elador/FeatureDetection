/*
 * CompleteExtendedHogFilter.cpp
 *
 *  Created on: 06.01.2014
 *      Author: poschmann
 */

#include "imageprocessing/CompleteExtendedHogFilter.hpp"
#include <stdexcept>

using cv::Mat;
using std::vector;
using std::invalid_argument;

namespace imageprocessing {

const float CompleteExtendedHogFilter::eps = 1e-4;

CompleteExtendedHogFilter::CompleteExtendedHogFilter(size_t cellSize, size_t binCount, bool signedGradients, bool unsignedGradients,
		bool interpolateBins, bool interpolateCells, float alpha) :
				cellSize(cellSize), binCount(binCount), signedGradients(signedGradients), unsignedGradients(unsignedGradients),
				interpolateBins(interpolateBins), interpolateCells(interpolateCells), alpha(alpha) {
	if (!signedGradients && !unsignedGradients)
		throw invalid_argument("CompleteExtendedHogFilter: signedGradients or unsignedGradients has to be true");
	if (signedGradients && unsignedGradients && binCount % 2 != 0)
		throw invalid_argument("CompleteExtendedHogFilter: if both signed and unsigned gradients should be used, the bin count has to be even");
	// build the look-up table for the bin informations given the gradients
	// index of the look-up table is the concatanation of the gradients of x and y (index = 512 * x + y)
	// value of the look-up table is the bin index and weight (or two bin indices and weights in case of bin interpolation)
	BinInformation binInformation;
	for (int x = 0; x < 512; ++x) {
		double gradientX = static_cast<double>(x - 256) / (2. * 255.);
		for (int y = 0; y < 512; ++y) {
			double gradientY = static_cast<double>(y - 256) / (2. * 255.);
			double direction = atan2(gradientY, gradientX);
			double magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);
			double binIndex;
			if (signedGradients) {
				direction += CV_PI;
				binIndex = direction * binCount / (2 * CV_PI);
			} else { // unsigned gradients
				if (direction < 0)
					direction += CV_PI;
				binIndex = direction * binCount / CV_PI;
			}
			if (interpolateBins) {
				binInformation.index1 = static_cast<int>(floor(binIndex)) % binCount;
				binInformation.index2 = static_cast<int>(ceil(binIndex)) % binCount;
				binInformation.weight2 = static_cast<float>(magnitude * (binIndex - floor(binIndex)));
				binInformation.weight1 = static_cast<float>(magnitude - binInformation.weight2);
			} else {
				binInformation.index1 = static_cast<int>(round(binIndex)) % binCount;
				binInformation.weight1 = static_cast<float>(magnitude);
				binInformation.index2 = binInformation.index1;
				binInformation.weight2 = 0;
			}
			binLut[512 * x + y] = binInformation;
		}
	}
}

Mat CompleteExtendedHogFilter::applyTo(const Mat& image, Mat& filtered) const {
	size_t cellRowCount = image.rows / cellSize;
	size_t cellColumnCount = image.cols / cellSize;
	size_t descriptorSize = binCount + (signedGradients && unsignedGradients ? binCount / 2 : 0) + 4;
	filtered = Mat::zeros(cellRowCount, cellColumnCount, CV_32FC(descriptorSize));
	buildInitialHistograms(filtered, image, cellRowCount, cellColumnCount);
	buildDescriptors(filtered, cellRowCount, cellColumnCount, descriptorSize);
	return filtered;
}

void CompleteExtendedHogFilter::createLut(vector<BinInformation>& lut, size_t size, size_t count) const {
	if (lut.size() != size) {
		lut.clear();
		lut.reserve(size);
		BinInformation entry;
		if (interpolateCells) {
			for (size_t matIndex = 0; matIndex < size; ++matIndex) {
				double realIndex = (static_cast<double>(matIndex) + 0.5) / static_cast<double>(cellSize) - 0.5;
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
				lut.push_back(entry);
			}
		} else {
			entry.index2 = -1;
			entry.weight1 = 1;
			entry.weight2 = 0;
			for (size_t matIndex = 0; matIndex < size; ++matIndex) {
				entry.index1 = matIndex / cellSize;
				lut.push_back(entry);
			}
		}
	}
}

void CompleteExtendedHogFilter::buildInitialHistograms(Mat& histograms, const Mat& image, size_t cellRowCount, size_t cellColumnCount) const {
	if (image.type() != CV_8UC1)
		throw invalid_argument("CompleteExtendedHogFilter: image must be of type CV_8UC1");

	createLut(rowLut, image.rows, cellRowCount);
	createLut(columnLut, image.cols, cellColumnCount);
	size_t height = cellRowCount * cellSize;
	size_t width = cellColumnCount * cellSize;

	for (size_t y = 0; y < height; ++y) {
		int rowIndex1 = rowLut[y].index1;
		int rowIndex2 = rowLut[y].index2;
		float rowWeight1 = rowLut[y].weight1;
		float rowWeight2 = rowLut[y].weight2;

		for (size_t x = 0; x < width; ++x) {
			int colIndex1 = columnLut[x].index1;
			int colIndex2 = columnLut[x].index2;
			float colWeight1 = columnLut[x].weight1;
			float colWeight2 = columnLut[x].weight2;

			int dx = image.at<uchar>(y, std::min(width - 1, x + 1)) - image.at<uchar>(y, std::max(0, static_cast<int>(x) - 1)) + 256;
			int dy = image.at<uchar>(std::min(height - 1, y + 1), x) - image.at<uchar>(std::max(0, static_cast<int>(y) - 1), x) + 256;
			const BinInformation& binInformation = binLut[dx * 512 + dy];

			if (interpolateCells) {
				float* histogram11Values = histograms.ptr<float>(rowIndex1, colIndex1);
				float* histogram12Values = histograms.ptr<float>(rowIndex1, colIndex2);
				float* histogram21Values = histograms.ptr<float>(rowIndex2, colIndex1);
				float* histogram22Values = histograms.ptr<float>(rowIndex2, colIndex2);
				if (interpolateBins) {
					histogram11Values[binInformation.index1] += binInformation.weight1 * rowWeight1 * colWeight1;
					histogram11Values[binInformation.index2] += binInformation.weight2 * rowWeight1 * colWeight1;
					histogram12Values[binInformation.index1] += binInformation.weight1 * rowWeight1 * colWeight2;
					histogram12Values[binInformation.index2] += binInformation.weight2 * rowWeight1 * colWeight2;
					histogram21Values[binInformation.index1] += binInformation.weight1 * rowWeight2 * colWeight1;
					histogram21Values[binInformation.index2] += binInformation.weight2 * rowWeight2 * colWeight1;
					histogram22Values[binInformation.index1] += binInformation.weight1 * rowWeight2 * colWeight2;
					histogram22Values[binInformation.index2] += binInformation.weight2 * rowWeight2 * colWeight2;
				} else {
					histogram11Values[binInformation.index1] += binInformation.weight1 * rowWeight1 * colWeight1;
					histogram12Values[binInformation.index1] += binInformation.weight1 * rowWeight1 * colWeight2;
					histogram21Values[binInformation.index1] += binInformation.weight1 * rowWeight2 * colWeight1;
					histogram22Values[binInformation.index1] += binInformation.weight1 * rowWeight2 * colWeight2;
				}
			} else {
				float* histogramValues = histograms.ptr<float>(rowIndex1, colIndex1);
				if (interpolateBins) {
					histogramValues[binInformation.index1] += binInformation.weight1;
					histogramValues[binInformation.index2] += binInformation.weight2;
				} else {
					histogramValues[binInformation.index1] += binInformation.weight1;
				}
			}
		}
	}
}

void CompleteExtendedHogFilter::buildDescriptors(Mat& descriptors, size_t cellRowCount, size_t cellColumnCount, size_t descriptorSize) const {
	size_t binHalfCount = binCount / 2;

	// compute gradient energy of cells
	Mat energies(descriptors.rows, descriptors.cols, CV_32FC1);
	if (signedGradients) {
		// gradient energy should be computed over unsigned gradients, so the signed parts have to be added up
		for (size_t rowIndex = 0; rowIndex < cellRowCount; ++rowIndex) {
			for (size_t colIndex = 0; colIndex < cellColumnCount; ++colIndex) {
				float energy = 0;
				float* histogramValues = descriptors.ptr<float>(rowIndex, colIndex);
				for (size_t binIndex = 0; binIndex < binHalfCount; ++binIndex) {
					float unsignedBinValue = histogramValues[binIndex] + histogramValues[binIndex + binHalfCount];
					energy += unsignedBinValue * unsignedBinValue;
				}
				energies.at<float>(rowIndex, colIndex) = energy;
			}
		}
	} else {
		for (size_t rowIndex = 0; rowIndex < cellRowCount; ++rowIndex) {
			for (size_t colIndex = 0; colIndex < cellColumnCount; ++colIndex) {
				float energy = 0;
				float* histogramValues = descriptors.ptr<float>(rowIndex, colIndex);
				for (size_t binIndex = 0; binIndex < binCount; ++binIndex)
					energy += histogramValues[binIndex] * histogramValues[binIndex];
				energies.at<float>(rowIndex, colIndex) = energy;
			}
		}
	}

	// build normalized cell descriptors
	if (signedGradients && unsignedGradients) { // signed and unsigned gradients should be combined into descriptor
		for (size_t rowIndex = 0; rowIndex < cellRowCount; ++rowIndex) {
			for (size_t colIndex = 0; colIndex < cellColumnCount; ++colIndex) {
				float* descriptor = descriptors.ptr<float>(rowIndex, colIndex);
				int r1 = rowIndex;
				int r0 = std::max(0, static_cast<int>(rowIndex) - 1);
				int r2 = std::min(rowIndex + 1, cellRowCount - 1);
				int c1 = colIndex;
				int c0 = std::max(0, static_cast<int>(colIndex) - 1);
				int c2 = std::min(colIndex + 1, cellColumnCount - 1);
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

				// unsigned orientation features (aka contrast-insensitive)
				for (size_t binIndex = 0; binIndex < binHalfCount; ++binIndex) {
					float sum = descriptor[binIndex] + descriptor[binIndex + binHalfCount];
					float h1 = std::min(alpha, sum * n1);
					float h2 = std::min(alpha, sum * n2);
					float h3 = std::min(alpha, sum * n3);
					float h4 = std::min(alpha, sum * n4);
					descriptor[binCount + binIndex] = 0.5 * (h1 + h2 + h3 + h4);
				}

				float t1 = 0;
				float t2 = 0;
				float t3 = 0;
				float t4 = 0;

				// signed orientation features (aka contrast-sensitive)
				for (size_t binIndex = 0; binIndex < binCount; ++binIndex) {
					float h1 = std::min(alpha, descriptor[binIndex] * n1);
					float h2 = std::min(alpha, descriptor[binIndex] * n2);
					float h3 = std::min(alpha, descriptor[binIndex] * n3);
					float h4 = std::min(alpha, descriptor[binIndex] * n4);
					descriptor[binIndex] = 0.5 * (h1 + h2 + h3 + h4);
					t1 += h1;
					t2 += h2;
					t3 += h3;
					t4 += h4;
				}

				// energy features
				descriptor[binCount + binHalfCount] = 0.2357 * t1;
				descriptor[binCount + binHalfCount + 1] = 0.2357 * t2;
				descriptor[binCount + binHalfCount + 2] = 0.2357 * t3;
				descriptor[binCount + binHalfCount + 3] = 0.2357 * t4;
			}
		}
	} else { // only signed or unsigned gradients should be in descriptor
		for (size_t rowIndex = 0; rowIndex < cellRowCount; ++rowIndex) {
			for (size_t colIndex = 0; colIndex < cellColumnCount; ++colIndex) {
				float* descriptor = descriptors.ptr<float>(rowIndex, colIndex);
				int r1 = rowIndex;
				int r0 = std::max(0, static_cast<int>(rowIndex) - 1);
				int r2 = std::min(rowIndex + 1, cellRowCount - 1);
				int c1 = colIndex;
				int c0 = std::max(0, static_cast<int>(colIndex) - 1);
				int c2 = std::min(colIndex + 1, cellColumnCount - 1);
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
				for (size_t binIndex = 0; binIndex < binCount; ++binIndex) {
					float h1 = std::min(alpha, descriptor[binIndex] * n1);
					float h2 = std::min(alpha, descriptor[binIndex] * n2);
					float h3 = std::min(alpha, descriptor[binIndex] * n3);
					float h4 = std::min(alpha, descriptor[binIndex] * n4);
					descriptor[binIndex] = 0.5 * (h1 + h2 + h3 + h4);
					t1 += h1;
					t2 += h2;
					t3 += h3;
					t4 += h4;
				}

				// energy features
				descriptor[binCount] = 0.2357 * t1;
				descriptor[binCount + 1] = 0.2357 * t2;
				descriptor[binCount + 2] = 0.2357 * t3;
				descriptor[binCount + 3] = 0.2357 * t4;
			}
		}
	}
}

} /* namespace imageprocessing */
