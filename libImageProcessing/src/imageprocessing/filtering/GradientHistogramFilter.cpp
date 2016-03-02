/*
 * GradientHistogramFilter.cpp
 *
 *  Created on: 11.12.2015
 *      Author: poschmann
 */

#include "imageprocessing/filtering/GradientHistogramFilter.hpp"
#include <stdexcept>
#include <vector>

using cv::Mat;
using cv::Vec2f;
using std::make_shared;
using std::invalid_argument;
using std::shared_ptr;
using std::vector;

namespace imageprocessing {
namespace filtering {

shared_ptr<GradientHistogramFilter> GradientHistogramFilter::full(
		int binCount, bool interpolate, int normalizationRadius, double normalizationConstant) {
	return make_shared<GradientHistogramFilter>(binCount, interpolate, false, true, true, normalizationRadius, normalizationConstant);
}

shared_ptr<GradientHistogramFilter> GradientHistogramFilter::half(
		int binCount, bool interpolate, int normalizationRadius, double normalizationConstant) {
	return make_shared<GradientHistogramFilter>(binCount, interpolate, true, false, true, normalizationRadius, normalizationConstant);
}

shared_ptr<GradientHistogramFilter> GradientHistogramFilter::both(
		int halfBinCount, bool interpolate, int normalizationRadius, double normalizationConstant) {
	return make_shared<GradientHistogramFilter>(2 * halfBinCount, interpolate, true, true, true, normalizationRadius, normalizationConstant);
}

GradientHistogramFilter::GradientHistogramFilter(
		int binCount, bool interpolate, bool half, bool full, bool magnitude, int normalizationRadius, double normalizationConstant) :
				magnitudeFilter(normalizationRadius, normalizationConstant),
				orientationFilter(full, normalizationRadius, normalizationConstant),
				fullAndHalf(half && full),
				magnitude(magnitude),
				binCount(binCount),
				halfBinCount(binCount / 2),
				descriptorSize(binCount + (fullAndHalf ? halfBinCount : 0) + (magnitude ? 1 : 0)),
				value2bin(binCount / orientationFilter.getUpperBound()),
				interpolate(interpolate) {
	if (binCount < 1)
		throw invalid_argument("GradientHistogramFilter: the binCount must be bigger than zero, but was " + std::to_string(binCount));
	if (!half && !full)
		throw invalid_argument("GradientHistogramFilter: either full or half (or both) must be true");
	if (half && full && binCount % 2 != 0)
		throw invalid_argument("GradientHistogramFilter: if both full and half histograms should be computed, the bin count must be even, but was "
				+ std::to_string(binCount));
	createGradientLut();
}

void GradientHistogramFilter::createGradientLut() {
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	// build the look-up table for gradient images of depth CV_8U
	// index of the look-up table is the binary concatenation of the gradients of x and y
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
			if (interpolate) {
				entry.fullBins = computeInterpolatedBins(orientation, 1);
				if (fullAndHalf) {
					entry.halfBins.bin1 = computeHalfBin(entry.fullBins.bin1);
					entry.halfBins.bin2 = computeHalfBin(entry.fullBins.bin2);
					entry.halfBins.weight1 = entry.fullBins.weight1;
					entry.halfBins.weight2 = entry.fullBins.weight2;
				}
			} else {
				entry.fullBins.bin1 = computeBin(orientation);
				entry.fullBins.weight1 = 1;
				if (fullAndHalf) {
					entry.halfBins.bin1 = computeHalfBin(entry.fullBins.bin1);
					entry.halfBins.weight1 = 1;
				}
			}
			binLut[gradientCode.index] = entry;
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

Mat GradientHistogramFilter::applyTo(const Mat& gradientImage, Mat& gradientHistogramImage) const {
	if (gradientImage.depth() != CV_8U && gradientImage.depth() != CV_32F)
		throw invalid_argument("GradientHistogramFilter: the gradient image depth must be CV_8U or CV_32F, but was "
				+ std::to_string(gradientImage.depth()));
	if (gradientImage.channels() % 2 != 0)
		throw invalid_argument("GradientHistogramFilter: the gradient image must have an even number of channels, but had "
				+ std::to_string(gradientImage.channels()));
	Mat singleGradientImage = reduceToStrongestGradient(gradientImage);
	Mat magnitudeImage = magnitudeFilter.applyTo(singleGradientImage);
	computeGradientHistogramImage(singleGradientImage, magnitudeImage, gradientHistogramImage);
	return gradientHistogramImage;
}

Mat GradientHistogramFilter::reduceToStrongestGradient(const Mat& gradientImage) const {
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

void GradientHistogramFilter::computeGradientHistogramImage(const Mat& singleGradientImage,
		const Mat& magnitudeImage, Mat& gradientHistogramImage) const {
	assert(singleGradientImage.channels() == 2);
	int magnitudeChannelIndex = magnitude ? descriptorSize - 1 : -1;
	gradientHistogramImage.create(singleGradientImage.rows, singleGradientImage.cols, CV_32FC(descriptorSize));
	if (singleGradientImage.depth() == CV_8U) {
		for (int row = 0; row < singleGradientImage.rows; ++row) {
			for (int col = 0; col < singleGradientImage.cols; ++col) {
				ushort gradientCode = singleGradientImage.at<ushort>(row, col); // concatenation of x gradient and y gradient (both uchar)
				float magnitude = magnitudeImage.at<float>(row, col);
				float* descriptor = gradientHistogramImage.ptr<float>(row, col);
				for (int ch = 0; ch < descriptorSize; ++ch)
					descriptor[ch] = 0;
				LutEntry entry = binLut[gradientCode];
				if (interpolate) {
					descriptor[entry.fullBins.bin1] = entry.fullBins.weight1 * magnitude;
					descriptor[entry.fullBins.bin2] = entry.fullBins.weight2 * magnitude;
					if (fullAndHalf) {
						descriptor[binCount + entry.halfBins.bin1] = entry.halfBins.weight1 * magnitude;
						descriptor[binCount + entry.halfBins.bin2] = entry.halfBins.weight2 * magnitude;
					}
				} else {
					descriptor[entry.fullBins.bin1] = magnitude;
					if (fullAndHalf) {
						descriptor[binCount + entry.halfBins.bin1] = magnitude;
					}
				}
				if (this->magnitude)
					descriptor[magnitudeChannelIndex] = magnitude;
			}
		}
	} else if (singleGradientImage.depth() == CV_32F) {
		for (int row = 0; row < singleGradientImage.rows; ++row) {
			for (int col = 0; col < singleGradientImage.cols; ++col) {
				const Vec2f& gradient = singleGradientImage.at<Vec2f>(row, col); // gradient for x and y
				float orientation = computeOrientation(gradient[0], gradient[1]);
				float magnitude = magnitudeImage.at<float>(row, col);
				float* descriptor = gradientHistogramImage.ptr<float>(row, col);
				for (int ch = 0; ch < descriptorSize; ++ch)
					descriptor[ch] = 0;
				if (interpolate) {
					Bins bins = computeInterpolatedBins(orientation, magnitude);
					descriptor[bins.bin1] = bins.weight1;
					descriptor[bins.bin2] = bins.weight2;
					if (fullAndHalf) {
						descriptor[binCount + computeHalfBin(bins.bin1)] = bins.weight1;
						descriptor[binCount + computeHalfBin(bins.bin2)] = bins.weight2;
					}
				} else {
					int bin = computeBin(orientation);
					descriptor[bin] = magnitude;
					if (fullAndHalf) {
						descriptor[binCount + computeHalfBin(bin)] = magnitude;
					}
				}
				if (this->magnitude)
					descriptor[magnitudeChannelIndex] = magnitude;
			}
		}
	}
}

float GradientHistogramFilter::computeOrientation(float gradientX, float gradientY) const {
	return orientationFilter.computeOrientation(gradientX, gradientY);
}

float GradientHistogramFilter::computeMagnitude(float gradientX, float gradientY) const {
	return magnitudeFilter.computeMagnitude(gradientX, gradientY);
}

float GradientHistogramFilter::computeSquaredMagnitude(float gradientX, float gradientY) const {
	return magnitudeFilter.computeSquaredMagnitude(gradientX, gradientY);
}

int GradientHistogramFilter::computeBin(float value) const {
	int bin = static_cast<int>(value * value2bin + 0.5f);
	if (bin == binCount)
		bin = 0;
	return bin;
}

// bin1 and bin2 are guaranteed to be different
GradientHistogramFilter::Bins GradientHistogramFilter::computeInterpolatedBins(float value, float weight) const {
	Bins bins;
	const float bin = value * value2bin;
	bins.bin1 = static_cast<int>(bin);
	bins.bin2 = bins.bin1 + 1;
	if (bins.bin2 == binCount)
		bins.bin2 = 0;
	bins.weight2 = weight * (bin - bins.bin1);
	bins.weight1 = weight - bins.weight2;
	return bins;
}

int GradientHistogramFilter::computeHalfBin(int bin) const {
	return bin < halfBinCount ? bin : bin - halfBinCount;
}

Mat GradientHistogramFilter::visualizeUnsignedHistograms(const Mat& descriptors, int unsignedBinCount, int offset, int cellSize) {
	if (descriptors.channels() < offset + unsignedBinCount)
		throw invalid_argument("GradientHistogramFilter: descriptors image must have at least ("
				+ std::to_string(offset) + " + " + std::to_string(unsignedBinCount)
				+ ") channels, but had only " + std::to_string(descriptors.channels()));
	vector<Mat> lines = drawLines(cellSize, unsignedBinCount);
	Mat floatViz = drawFloatVisualization(descriptors, lines, unsignedBinCount, offset);
	return rescaleToUchar(floatViz, descriptors, unsignedBinCount, offset);
}

vector<Mat> GradientHistogramFilter::drawLines(int cellSize, int unsignedBinCount) {
	vector<Mat> bars(unsignedBinCount);
	int cellHalfSize = cellSize / 2;
	bool cellSizeIsEven = cellSize % 2 == 0;
	int start = cellSizeIsEven ? 2 : 1;
	int end = cellSize - 2;
	bars[0] = Mat::zeros(cellSize, cellSize, CV_32FC1);
	cv::line(bars[0], cv::Point(cellHalfSize, start), cv::Point(cellHalfSize, end), cv::Scalar(1.0), 1);
	cv::Point2f center(cellHalfSize, cellHalfSize);
	for (int bin = 1; bin < unsignedBinCount; ++bin) {
		double angle = bin * 180 / unsignedBinCount; // positive angle is clockwise, because y-axis points downwards
		Mat rotation = cv::getRotationMatrix2D(center, -angle, 1.0); // expects positive angle to be counter-clockwise
		cv::warpAffine(bars[0], bars[bin], rotation, bars[0].size(), cv::INTER_CUBIC);
	}
	return bars;
}

Mat GradientHistogramFilter::drawFloatVisualization(const Mat& descriptors, const vector<Mat>& lines, int unsignedBinCount, int offset) {
	int cellSize = lines[0].rows;
	Mat floatViz = Mat::zeros(cellSize * descriptors.rows, cellSize * descriptors.cols, CV_32FC1);
	for (int row = 0; row < descriptors.rows; ++row) {
		for (int col = 0; col < descriptors.cols; ++col) {
			const float* descriptor = descriptors.ptr<float>(row, col);
			Mat cell(floatViz, cv::Rect(cellSize * col, cellSize * row, cellSize, cellSize));
			mergeWeightedLines(cell, descriptor, lines, unsignedBinCount, offset);
		}
	}
	return floatViz;
}

void GradientHistogramFilter::mergeWeightedLines(Mat& cell, const float* descriptor, const vector<Mat>& lines, int unsignedBinCount, int offset) {
	for (int bin = 0; bin < unsignedBinCount; ++bin) {
		float weight = getWeight(descriptor, bin, offset);
		if (weight > 0)
			cell = cv::max(cell, weight * lines[bin]);
	}
}

Mat GradientHistogramFilter::rescaleToUchar(const Mat& floatViz, const Mat& descriptors, int unsignedBinCount, int offset) {
	Mat visualization;
	floatViz.convertTo(visualization, CV_8U, 255.0 / getMaxWeight(descriptors, unsignedBinCount, offset));
	return visualization;
}

float GradientHistogramFilter::getMaxWeight(const Mat& descriptors, int unsignedBinCount, int offset) {
	float maxWeight = 0;
	for (int row = 0; row < descriptors.rows; ++row) {
		for (int col = 0; col < descriptors.cols; ++col) {
			const float* descriptor = descriptors.ptr<float>(row, col);
			for (int bin = 0; bin < unsignedBinCount; ++bin) {
				float weight = getWeight(descriptor, bin, offset);
				if (weight < 0)
					weight = -weight;
				maxWeight = std::max(maxWeight, weight);
			}
		}
	}
	return maxWeight;
}

float GradientHistogramFilter::getWeight(const float* descriptor, int bin, int offset) {
	return descriptor[offset + bin];
}

} /* namespace filtering */
} /* namespace imageprocessing */
