/*
 * FhogFilter.cpp
 *
 *  Created on: 24.02.2016
 *      Author: poschmann
 */

#include "imageprocessing/filtering/FhogFilter.hpp"
#include <stdexcept>
#include <vector>

using cv::Mat;
using cv::Vec2f;
using std::invalid_argument;
using std::vector;

namespace imageprocessing {
namespace filtering {

FhogFilter::FhogFilter(int cellSize, int unsignedBinCount, bool interpolateBins, bool interpolateCells, float alpha) :
		magnitudeFilter(0),
		orientationFilter(true, 0),
		fhogAggregationFilter(cellSize, interpolateCells, alpha),
		cellSize(cellSize),
		signedBinCount(2 * unsignedBinCount),
		unsignedBinCount(unsignedBinCount),
		value2bin(signedBinCount / orientationFilter.getUpperBound()),
		interpolateBins(interpolateBins),
		interpolateCells(interpolateCells) {
	if (unsignedBinCount < 1)
		throw invalid_argument("FhogFilter: unsignedBinCount must be bigger than zero, but was: " + std::to_string(unsignedBinCount));
	createGradientLut();
}

void FhogFilter::createGradientLut() {
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	// build the look-up table for gradient images of depth CV_8U
	// index of the look-up table is the binary concatanation of the gradients of x and y
	// values inside the look-up table are the bin indices and weights
	gradientCode.gradient.x = 0;
	for (int x = 0; x < 256; ++x) {
		float gradientX = (x - 127.f) / 255.f;
		gradientCode.gradient.y = 0;
		for (int y = 0; y < 256; ++y) {
			float gradientY = (y - 127.f) / 255.f;
			LutEntry entry;
			entry.magnitude = computeMagnitude(gradientX, gradientY);
			float orientation = computeOrientation(gradientX, gradientY);
			if (interpolateBins) {
				entry.bins = computeInterpolatedBins(orientation, entry.magnitude);
			} else {
				entry.bins.index1 = computeBin(orientation);
				entry.bins.weight1 = entry.magnitude;
			}
			binLut[gradientCode.index] = entry;
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

Mat FhogFilter::applyTo(const Mat& gradientImage, Mat& descriptors) const {
	if (gradientImage.depth() != CV_8U && gradientImage.depth() != CV_32F)
		throw invalid_argument("FhogFilter: the gradient image depth must be CV_8U or CV_32F, but was "
				+ std::to_string(gradientImage.depth()));
	if (gradientImage.channels() % 2 != 0)
		throw invalid_argument("FhogFilter: the gradient image must have an even number of channels, but had "
				+ std::to_string(gradientImage.channels()));
	Mat singleGradientImage = reduceToStrongestGradient(gradientImage);
	int rows = gradientImage.rows / cellSize;
	int cols = gradientImage.cols / cellSize;
	int descriptorSize = signedBinCount + unsignedBinCount + 4;
	descriptors = Mat::zeros(rows, cols, CV_32FC(descriptorSize));
	computeSignedHistograms(singleGradientImage, descriptors);
	fhogAggregationFilter.computeDescriptors(descriptors, descriptors, signedBinCount);
	return descriptors;
}

Mat FhogFilter::reduceToStrongestGradient(const Mat& gradientImage) const {
	if (gradientImage.channels() == 2)
		return gradientImage;
	Mat singleGradientImage(gradientImage.rows, gradientImage.cols, CV_MAKETYPE(gradientImage.depth(), 2));
	int originalChannels = gradientImage.channels() / 2;
	if (gradientImage.depth() == CV_8U) {
		for (int row = 0; row < gradientImage.rows; ++row) {
			for (int col = 0; col < gradientImage.cols; ++col) {
				const ushort* gradientCodes = gradientImage.ptr<ushort>(row, col); // concatenation of x gradient and y gradient (both uchar)
				ushort strongestGradientCode = gradientCodes[0];
				float strongestMagnitude = binLut[strongestGradientCode].magnitude;
				for (int ch = 1; ch < originalChannels; ++ch) {
					ushort gradientCode = gradientCodes[ch];
					if (strongestMagnitude < binLut[gradientCode].magnitude) {
						strongestGradientCode = gradientCode;
						strongestMagnitude = binLut[gradientCode].magnitude;
					}
				}
				singleGradientImage.at<ushort>(row, col) = strongestGradientCode;
			}
		}
	} else if (gradientImage.depth() == CV_32F) {
		for (int row = 0; row < gradientImage.rows; ++row) {
			for (int col = 0; col < gradientImage.cols; ++col) {
				const Vec2f* gradients = gradientImage.ptr<Vec2f>(row, col); // gradient for x and y
				Vec2f strongestGradient = gradients[0];
				float strongestSquaredMagnitude = computeSquaredMagnitude(strongestGradient[0], strongestGradient[1]);
				for (int ch = 1; ch < originalChannels; ++ch) {
					const Vec2f& gradient = gradients[ch];
					float squaredMagnitude = computeSquaredMagnitude(gradient[0], gradient[1]);
					if (squaredMagnitude > strongestSquaredMagnitude) {
						strongestSquaredMagnitude = squaredMagnitude;
						strongestGradient = gradient;
					}
				}
				singleGradientImage.at<Vec2f>(row, col) = strongestGradient;
			}
		}
	}
	return singleGradientImage;
}

void FhogFilter::computeSignedHistograms(const Mat& gradients, Mat& signedHistograms) const {
	assert(gradients.channels() == 2);
	assert(gradients.rows >= signedHistograms.rows * cellSize);
	assert(gradients.cols >= signedHistograms.cols * cellSize);
	vector<Coefficients> rowCoefficients = computeInterpolationCoefficents(signedHistograms.rows * cellSize, signedHistograms.rows);
	vector<Coefficients> colCoefficients = computeInterpolationCoefficents(signedHistograms.cols * cellSize, signedHistograms.cols);
	if (gradients.depth() == CV_8U) {
		for (int gradientRow = 0; gradientRow < rowCoefficients.size(); ++gradientRow) {
			int histogramRow1 = rowCoefficients[gradientRow].index1;
			int histogramRow2 = rowCoefficients[gradientRow].index2;
			float rowWeight1 = rowCoefficients[gradientRow].weight1;
			float rowWeight2 = rowCoefficients[gradientRow].weight2;

			for (int gradientCol = 0; gradientCol < colCoefficients.size(); ++gradientCol) {
				int histogramCol1 = colCoefficients[gradientCol].index1;
				int histogramCol2 = colCoefficients[gradientCol].index2;
				float colWeight1 = colCoefficients[gradientCol].weight1;
				float colWeight2 = colCoefficients[gradientCol].weight2;

				ushort gradientCode = gradients.at<ushort>(gradientRow, gradientCol); // concatenation of x gradient and y gradient (both uchar)
				const LutEntry& entry = binLut[gradientCode];
				if (interpolateCells) {
					float* histogram11 = signedHistograms.ptr<float>(histogramRow1, histogramCol1);
					float* histogram12 = signedHistograms.ptr<float>(histogramRow1, histogramCol2);
					float* histogram21 = signedHistograms.ptr<float>(histogramRow2, histogramCol1);
					float* histogram22 = signedHistograms.ptr<float>(histogramRow2, histogramCol2);
					if (interpolateBins) {
						histogram11[entry.bins.index1] += entry.bins.weight1 * rowWeight1 * colWeight1;
						histogram11[entry.bins.index2] += entry.bins.weight2 * rowWeight1 * colWeight1;
						histogram12[entry.bins.index1] += entry.bins.weight1 * rowWeight1 * colWeight2;
						histogram12[entry.bins.index2] += entry.bins.weight2 * rowWeight1 * colWeight2;
						histogram21[entry.bins.index1] += entry.bins.weight1 * rowWeight2 * colWeight1;
						histogram21[entry.bins.index2] += entry.bins.weight2 * rowWeight2 * colWeight1;
						histogram22[entry.bins.index1] += entry.bins.weight1 * rowWeight2 * colWeight2;
						histogram22[entry.bins.index2] += entry.bins.weight2 * rowWeight2 * colWeight2;
					} else {
						histogram11[entry.bins.index1] += entry.bins.weight1 * rowWeight1 * colWeight1;
						histogram12[entry.bins.index1] += entry.bins.weight1 * rowWeight1 * colWeight2;
						histogram21[entry.bins.index1] += entry.bins.weight1 * rowWeight2 * colWeight1;
						histogram22[entry.bins.index1] += entry.bins.weight1 * rowWeight2 * colWeight2;
					}
				} else {
					float* histogram = signedHistograms.ptr<float>(histogramRow1, histogramCol1);
					if (interpolateBins) {
						histogram[entry.bins.index1] += entry.bins.weight1;
						histogram[entry.bins.index2] += entry.bins.weight2;
					} else {
						histogram[entry.bins.index1] += entry.bins.weight1;
					}
				}
			}
		}
	} else if (gradients.depth() == CV_32F) {
		for (int gradientRow = 0; gradientRow < rowCoefficients.size(); ++gradientRow) {
			int histogramRow1 = rowCoefficients[gradientRow].index1;
			int histogramRow2 = rowCoefficients[gradientRow].index2;
			float rowWeight1 = rowCoefficients[gradientRow].weight1;
			float rowWeight2 = rowCoefficients[gradientRow].weight2;

			for (int gradientCol = 0; gradientCol < colCoefficients.size(); ++gradientCol) {
				int histogramCol1 = colCoefficients[gradientCol].index1;
				int histogramCol2 = colCoefficients[gradientCol].index2;
				float colWeight1 = colCoefficients[gradientCol].weight1;
				float colWeight2 = colCoefficients[gradientCol].weight2;

				const Vec2f& gradient = gradients.at<Vec2f>(gradientRow, gradientCol); // gradient for x and y
				float orientation = computeOrientation(gradient[0], gradient[1]);
				float magnitude = computeMagnitude(gradient[0], gradient[1]);
				if (interpolateCells) {
					float* histogram11 = signedHistograms.ptr<float>(histogramRow1, histogramCol1);
					float* histogram12 = signedHistograms.ptr<float>(histogramRow1, histogramCol2);
					float* histogram21 = signedHistograms.ptr<float>(histogramRow2, histogramCol1);
					float* histogram22 = signedHistograms.ptr<float>(histogramRow2, histogramCol2);
					if (interpolateBins) {
						Coefficients bins = computeInterpolatedBins(orientation, magnitude);
						histogram11[bins.index1] += bins.weight1 * rowWeight1 * colWeight1;
						histogram11[bins.index2] += bins.weight2 * rowWeight1 * colWeight1;
						histogram12[bins.index1] += bins.weight1 * rowWeight1 * colWeight2;
						histogram12[bins.index2] += bins.weight2 * rowWeight1 * colWeight2;
						histogram21[bins.index1] += bins.weight1 * rowWeight2 * colWeight1;
						histogram21[bins.index2] += bins.weight2 * rowWeight2 * colWeight1;
						histogram22[bins.index1] += bins.weight1 * rowWeight2 * colWeight2;
						histogram22[bins.index2] += bins.weight2 * rowWeight2 * colWeight2;
					} else {
						int bin = computeBin(orientation);
						histogram11[bin] += magnitude * rowWeight1 * colWeight1;
						histogram12[bin] += magnitude * rowWeight1 * colWeight2;
						histogram21[bin] += magnitude * rowWeight2 * colWeight1;
						histogram22[bin] += magnitude * rowWeight2 * colWeight2;
					}
				} else {
					float* histogram = signedHistograms.ptr<float>(histogramRow1, histogramCol1);
					if (interpolateBins) {
						Coefficients bins = computeInterpolatedBins(orientation, magnitude);
						histogram[bins.index1] += bins.weight1;
						histogram[bins.index2] += bins.weight2;
					} else {
						int bin = computeBin(orientation);
						histogram[bin] += magnitude;
					}
				}
			}
		}
	}
}

vector<FhogFilter::Coefficients> FhogFilter::computeInterpolationCoefficents(int sizeInPixels, int sizeInCells) const {
	vector<Coefficients> coefficients(sizeInPixels);
	if (interpolateCells) {
		for (int pixel = 0; pixel < sizeInPixels; ++pixel) {
			float realCellIndex = (pixel + 0.5f) / cellSize - 0.5f;
			int index1 = static_cast<int>(std::floor(realCellIndex));
			int index2 = index1 + 1;
			float weight2 = realCellIndex - index1;
			float weight1 = index2 - realCellIndex;
			if (index1 < 0) {
				index1 = index2;
				weight1 = 0;
			} else if (index2 >= sizeInCells) {
				index2 = index1;
				weight2 = 0;
			}
			coefficients[pixel] = { index1, index2, weight1, weight2 };
		}
	} else {
		for (int pixel = 0; pixel < sizeInPixels; ++pixel) {
			coefficients[pixel] = { pixel / cellSize, -1, 1, 0 };
		}
	}
	return coefficients;
}

float FhogFilter::computeOrientation(float gradientX, float gradientY) const {
	return orientationFilter.computeOrientation(gradientX, gradientY);
}

float FhogFilter::computeMagnitude(float gradientX, float gradientY) const {
	return magnitudeFilter.computeMagnitude(gradientX, gradientY);
}

float FhogFilter::computeSquaredMagnitude(float gradientX, float gradientY) const {
	return magnitudeFilter.computeSquaredMagnitude(gradientX, gradientY);
}

int FhogFilter::computeBin(float value) const {
	int bin = static_cast<int>(value * value2bin + 0.5f);
	if (bin == signedBinCount)
		bin = 0;
	return bin;
}

// bin1 and bin2 are guaranteed to be different
FhogFilter::Coefficients FhogFilter::computeInterpolatedBins(float value, float weight) const {
	Coefficients bins;
	const float bin = value * value2bin;
	bins.index1 = static_cast<int>(bin);
	bins.index2 = bins.index1 + 1;
	if (bins.index2 == signedBinCount)
		bins.index2 = 0;
	bins.weight2 = weight * (bin - bins.index1);
	bins.weight1 = weight - bins.weight2;
	return bins;
}

cv::Mat FhogFilter::visualizeUnsignedHistograms(const cv::Mat& descriptors, int cellSize) {
	return FhogAggregationFilter::visualizeUnsignedHistograms(descriptors, cellSize);
}

} /* namespace filtering */
} /* namespace imageprocessing */
