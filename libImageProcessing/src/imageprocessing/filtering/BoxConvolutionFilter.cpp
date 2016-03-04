/*
 * BoxConvolutionFilter.cpp
 *
 *  Created on: 26.01.2016
 *      Author: poschmann
 */

#include "imageprocessing/filtering/BoxConvolutionFilter.hpp"
#include <stdexcept>
#include <vector>

using cv::Mat;
using std::invalid_argument;
using std::to_string;
using std::vector;

namespace imageprocessing {
namespace filtering {

BoxConvolutionFilter::BoxConvolutionFilter(int size, int downScaling, float alpha, float delta) :
		odd(size % 2 != 0),
		size(size),
		radius(size / 2),
		downScaleFactor(downScaling),
		downScaleOffset(odd ? downScaleFactor / 2 : downScaleFactor / 2 - 1),
		noDownScaling(downScaleFactor == 1),
		normalizer(alpha / (size * size)),
		delta(delta / normalizer) {
	if (size < 1)
		throw invalid_argument("BoxConvolutionFilter: size must be greater than zero, but was " + to_string(size));
	if (downScaling < 1)
		throw invalid_argument("BoxConvolutionFilter: downScaling must be greater than zero, but was " + to_string(downScaling));
}

// the implementation is based on the observation that the difference of two subsequent filtered values
// of a one-dimensional box convolution differs only in two values, independent of the filter size
//
// consider for example the following one-dimensional values: a b c d e f g
// with the following filter kernel: (1 1 1 1)/4
//
//   | filtered value | difference
// a | a + a + b + c  |(-a +c)
// b | a + b + c + d  | -a +d
// c | b + c + d + e  | -a +e
// d | c + d + e + f  | -b +f
// e | d + e + f + g  | -c +g
// f | e + f + g + g  | -d +g
// g | f + g + g + g  | -e +g
//
// the difference depends on just two values, regardless of filter size
// this allows an implementation of the filtering whose processing time is approximately constant per pixel

Mat BoxConvolutionFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.depth() != CV_32F)
		throw invalid_argument("BoxConvolutionFilter: image must have a depth of CV_32F, but was " + to_string(image.depth()));
	if (image.rows <= radius)
		throw invalid_argument("BoxConvolutionFilter: image must have at least "
				+ to_string(radius + 1) + " rows, but had only " + to_string(image.rows));
	if (image.cols <= radius)
		throw invalid_argument("BoxConvolutionFilter: image must have at least "
				+ to_string(radius + 1) + " columns, but had only " + to_string(image.cols));
	filtered.create(image.rows / downScaleFactor, image.cols / downScaleFactor, image.type());
	int channels = image.channels();

	int valuesPerRow = image.cols * channels;
	vector<float> intermediate(valuesPerRow);
	// initialize intermediate result (vertical filtering) of first row
	if (odd) {
		const float* channelValues = image.ptr<float>(0);
		for (int i = 0; i < valuesPerRow; ++i)
			intermediate[i] = channelValues[i];
	}
	for (int i = 0; i < radius; ++i) {
		const float* firstRowChannelValues = image.ptr<float>(0);
		const float* rowChannelValues = image.ptr<float>(i + 1);
		for (int k = 0; k < valuesPerRow; ++k)
			intermediate[k] += firstRowChannelValues[k] + rowChannelValues[k];
	}
	// filter intermediate filtering results (first row) horizontally
	int outputRow = 0;
	if (shouldWriteToOutput(0)) {
		filterRow(intermediate.data(), filtered.ptr<float>(0), image.cols, channels);
		outputRow = 1;
	}

	// filter remaining rows
	int subOffset = (size + 1) / 2;
	int addOffset = size / 2;
	int lastInputRow = image.rows - 1;
	int inputRows = (filtered.rows - 1) * downScaleFactor + downScaleOffset + 1;
	for (int inputRow = 1; inputRow < inputRows; ++inputRow) {
		// compute intermediate result (vertical filtering) of row
		int subIndex = std::max(           0, inputRow - subOffset);
		int addIndex = std::min(lastInputRow, inputRow + addOffset);
		const float* subValues = image.ptr<float>(subIndex);
		const float* addValues = image.ptr<float>(addIndex);
		for (int i = 0; i < valuesPerRow; ++i)
			intermediate[i] += addValues[i] - subValues[i];
		// filter intermediate result (row) horizontally
		if (shouldWriteToOutput(inputRow)) {
			filterRow(intermediate.data(), filtered.ptr<float>(outputRow), image.cols, channels);
			++outputRow;
		}
	}
	return filtered;
}

void BoxConvolutionFilter::filterRow(const float* inputValues, float* outputValues, int cols, int channels) const {
	assert(cols > radius);
	ConstRow input{inputValues, cols, channels};
	Row output{outputValues, cols / downScaleFactor, channels};

	for (int ch = 0; ch < channels; ++ch) {
		// initialize first output value
		float outputValue = delta + (odd ? input.value(0, ch) : 0);
		for (int col = 0; col < radius; ++col)
			outputValue += input.value(0, ch) + input.value(col + 1, ch);
		// write first output value
		int outputCol = 0;
		if (shouldWriteToOutput(0)) {
			output.value(0, ch) = normalizer * outputValue;
			outputCol = 1;
		}

		// filter remaining values
		int subOffset = (size + 1) / 2;
		int addOffset = size / 2;
		int lastInputCol = cols - 1;
		int inputCols = (output.cols - 1) * downScaleFactor + downScaleOffset + 1;
		for (int inputCol = 1; inputCol < subOffset; ++inputCol) {
			int subCol = 0;
			int addCol = inputCol + addOffset;
			filterNextValue(input, output, ch, inputCol, outputCol,
					addCol, subCol, outputValue);
		}
		for (int inputCol = subOffset; inputCol < std::min(inputCols, cols - addOffset); ++inputCol) {
			int subCol = inputCol - subOffset;
			int addCol = inputCol + addOffset;
			filterNextValue(input, output, ch, inputCol, outputCol,
					addCol, subCol, outputValue);
		}
		for (int inputCol = cols - addOffset; inputCol < inputCols; ++inputCol) {
			int subCol = inputCol - subOffset;
			int addCol = lastInputCol;
			filterNextValue(input, output, ch, inputCol, outputCol,
					addCol, subCol, outputValue);
		}
	}
}

void BoxConvolutionFilter::filterNextValue(ConstRow input, Row output, int ch,
		int inputCol, int& outputCol, int addCol, int subCol, float& outputValue) const {
	outputValue += input.value(addCol, ch) - input.value(subCol, ch);
	if (shouldWriteToOutput(inputCol)) {
		output.value(outputCol, ch) = normalizer * outputValue;
		++outputCol;
	}
}

bool BoxConvolutionFilter::shouldWriteToOutput(int index) const {
	return noDownScaling || (index - downScaleOffset) % downScaleFactor == 0;
}

} /* namespace filtering */
} /* namespace imageprocessing */
