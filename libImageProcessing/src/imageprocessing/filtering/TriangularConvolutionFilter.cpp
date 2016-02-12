/*
 * TriangularConvolutionFilter.cpp
 *
 *  Created on: 19.01.2016
 *      Author: poschmann
 */

#include "imageprocessing/filtering/TriangularConvolutionFilter.hpp"
#include <stdexcept>
#include <vector>

using cv::Mat;
using std::invalid_argument;
using std::to_string;
using std::vector;

namespace imageprocessing {
namespace filtering {

TriangularConvolutionFilter::TriangularConvolutionFilter(int size, int downScaling, float alpha, float delta) :
		odd(size % 2 != 0),
		size(size),
		radius(size / 2),
		downScaleFactor(downScaling),
		downScaleOffset(odd ? downScaleFactor / 2 : downScaleFactor / 2 - 1),
		noDownScaling(downScaleFactor == 1),
		normalizer(alpha / (odd ? ((radius + 1) * (radius + 1) * (radius + 1) * (radius + 1)) : ((radius + 1) * (radius + 1) * radius * radius))),
		delta(delta / normalizer) {
	if (size < 1)
		throw invalid_argument("TriangularConvolutionFilter: size must be greater than zero, but was " + to_string(size));
	if (downScaling < 1)
		throw invalid_argument("TriangularConvolutionFilter: downScaling must be greater than zero, but was " + to_string(downScaling));
}

// the implementation is based on the observation that the difference of the difference between two subsequent filtered values
// of a one-dimensional triangular convolution differs only in four (or three) values, independent of the filter size
//
// consider for example the following one-dimensional values: a b c d e f g
//
// with the following filter kernel of odd size: (1 2 3 2 1)/9
//
//   | filtered value       | 1st order difference | 2nd order difference
// a | a + 2a + 3a + 2b + c |(-a -a -a +a +b +c)   |
// b | a + 2a + 3b + 2c + d | -a -a -a +b +c +d    |(+a -a -a +d)
// c | a + 2b + 3c + 2d + e | -a -a -b +c +d +e    | +a -b -b +e
// d | b + 2c + 3d + 2e + f | -a -b -c +d +e +f    | +a -c -c +f
// e | c + 2d + 3e + 2f + g | -b -c -d +e +f +g    | +a -d -d +g
// f | d + 2e + 3f + 2g + g | -c -d -e +f +g +g    | +b -e -e +g
// g | e + 2f + 3g + 2g + g | -d -e -f +g +g +g    | +c -f -f +g
//
// or with the following filter kernel of even size: (1 2 2 1)/6
//
//   | filtered value  | 1st order diff | 2nd order diff
// a | a + 2a + 2b + c |(-a -a +b +c)   |
// b | a + 2b + 2c + d | -a -a +c +d    |(+a -a -b +d)
// c | b + 2c + 2d + e | -a -b +d +e    | +a -b -c +e
// d | c + 2d + 2e + f | -b -c +e +f    | +a -c -d +f
// e | d + 2e + 2f + g | -c -d +f +g    | +a -d -e +g
// f | e + 2f + 2g + g | -d -e +g +g    | +b -e -f +g
// g | f + 2g + 2g + g | -e -f +g +g    | +c -f -g +g
//
// in both cases, the first order difference depends on the filter size, but the second order difference does not.
// it only depends on four (even size) or three (odd size) values
// this allows an implementation of the filtering whose processing time is approximately constant per pixel

Mat TriangularConvolutionFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.depth() != CV_32F)
		throw invalid_argument("TriangularConvolutionFilter: image must have a depth of CV_32F, but was " + to_string(image.depth()));
	if (image.rows <= radius)
		throw invalid_argument("TriangularConvolutionFilter: image must have at least "
				+ to_string(radius + 1) + " rows, but had only " + to_string(image.rows));
	if (image.cols < size + 1)
		throw invalid_argument("TriangularConvolutionFilter: image must have at least "
				+ to_string(size + 1) + " columns, but had only " + to_string(image.cols));
	filtered.create(image.rows / downScaleFactor, image.cols / downScaleFactor, image.type());

	int channels = image.channels();
	int valuesPerRow = image.cols * channels;
	vector<float> difference(valuesPerRow);
	vector<float> intermediate(valuesPerRow);
	// initialize intermediate result (vertical filtering) of first row
	if (odd) {
		const float* channelValues = image.ptr<float>(0);
		for (int i = 0; i < valuesPerRow; ++i) {
			difference[i] = channelValues[i];
			intermediate[i] = difference[i];
		}
	}
	for (int i = 0; i < radius; ++i) {
		const float* firstRowChannelValues = image.ptr<float>(0);
		const float* rowChannelValues = image.ptr<float>(i + 1);
		for (int k = 0; k < valuesPerRow; ++k) {
			difference[k] += firstRowChannelValues[k] + rowChannelValues[k];
			intermediate[k] += difference[k];
		}
	}
	// initialize difference of intermediate result
	for (int i = 0; i < difference.size(); ++i)
		difference[i] = 0;
	for (int i = 0; i < radius; ++i) {
		const float* firstRowChannelValues = image.ptr<float>(0);
		const float* rowChannelValues = image.ptr<float>(i + 1);
		for (int k = 0; k < valuesPerRow; ++k) {
			difference[k] += rowChannelValues[k] - firstRowChannelValues[k];
		}
	}
	// filter intermediate filtering results (first row) horizontally
	int outputRow = 0;
	if (isRelevantForOutput(0)) {
		filterRow(intermediate.data(), filtered.ptr<float>(0), image.cols, channels);
		outputRow = 1;
	}

	// filter remaining rows
	int lastInputRow = image.rows - 1;
	int inputRows = (filtered.rows - 1) * downScaleFactor + downScaleOffset + 1;
	int addOffset1 = (size + 1) / 2 + 1;
	int subOffset1 = 1;
	int subOffset2 = (size + 1) / 2 - size / 2; // even size: 0  odd size: 1
	int addOffset2 = size / 2;
	for (int inputRow = 1; inputRow < inputRows; ++inputRow) {
		// compute intermediate result (vertical filtering) of row
		int addIndex1 = std::max(           0, inputRow - addOffset1);
		int subIndex1 =                        inputRow - subOffset1;
		int subIndex2 =                        inputRow - subOffset2;
		int addIndex2 = std::min(lastInputRow, inputRow + addOffset2);
		const float* addValues1 = image.ptr<float>(addIndex1);
		const float* addValues2 = image.ptr<float>(addIndex2);
		const float* subValues1 = image.ptr<float>(subIndex1);
		const float* subValues2 = image.ptr<float>(subIndex2);
		for (int i = 0; i < valuesPerRow; ++i) {
			difference[i] += addValues1[i] + addValues2[i] - subValues1[i] - subValues2[i];
			intermediate[i] += difference[i];
		}
		// filter intermediate result (row) horizontally
		if (isRelevantForOutput(inputRow)) {
			filterRow(intermediate.data(), filtered.ptr<float>(outputRow), image.cols, channels);
			++outputRow;
		}
	}
	return filtered;
}

void TriangularConvolutionFilter::filterRow(const float* inputValues, float* outputValues, int cols, int channels) const {
	assert(cols >= size + 1);
	ConstRow input{inputValues, cols, channels};
	Row output{outputValues, cols / downScaleFactor, channels};

	for (int ch = 0; ch < input.channels; ++ch) {
		// initialize first output value
		float outputDifference = odd ? input.value(0, ch) : 0;
		float outputValue = odd ? (delta + outputDifference) : delta;
		for (int col = 0; col < radius; ++col) {
			outputDifference += input.value(0, ch) + input.value(col + 1, ch);
			outputValue += outputDifference;
		}
		// initialize difference of output value
		outputDifference = 0;
		for (int col = 0; col < radius; ++col) {
			outputDifference += input.value(col + 1, ch) - input.value(0, ch);
		}
		// write first output value
		int outputCol = 0;
		if (isRelevantForOutput(0)) {
			output.value(0, ch) = normalizer * outputValue;
			outputCol = 1;
		}

		// filter remaining values
		int lastInputCol = input.cols - 1;
		int inputCols = (output.cols - 1) * downScaleFactor + downScaleOffset + 1;
		int addOffset1 = (size + 1) / 2 + 1;
		int subOffset1 = 1;
		int subOffset2 = (size + 1) / 2 - size / 2; // even size: 0  odd size: 1
		int addOffset2 = size / 2;
		for (int inputCol = 1; inputCol < addOffset1; ++inputCol) {
			int addCol1 = 0;
			int subCol1 = inputCol - subOffset1;
			int subCol2 = inputCol - subOffset2;
			int addCol2 = inputCol + addOffset2;
			filterNextValue(input, output, ch, inputCol, outputCol,
					addCol1, addCol2, subCol1, subCol2, outputDifference, outputValue);
		}
		for (int inputCol = addOffset1; inputCol < std::min(inputCols, input.cols - addOffset2); ++inputCol) {
			int addCol1 = inputCol - addOffset1;
			int subCol1 = inputCol - subOffset1;
			int subCol2 = inputCol - subOffset2;
			int addCol2 = inputCol + addOffset2;
			filterNextValue(input, output, ch, inputCol, outputCol,
					addCol1, addCol2, subCol1, subCol2, outputDifference, outputValue);
		}
		for (int inputCol = input.cols - addOffset2; inputCol < inputCols; ++inputCol) {
			int addCol1 = inputCol - addOffset1;
			int subCol1 = inputCol - subOffset1;
			int subCol2 = inputCol - subOffset2;
			int addCol2 = lastInputCol;
			filterNextValue(input, output, ch, inputCol, outputCol,
					addCol1, addCol2, subCol1, subCol2, outputDifference, outputValue);
		}
	}
}

void TriangularConvolutionFilter::filterNextValue(ConstRow input, Row output, int ch, int inputCol, int& outputCol,
		int addCol1, int addCol2, int subCol1, int subCol2, float& outputDifference, float& outputValue) const {
	outputDifference += input.value(addCol1, ch) + input.value(addCol2, ch) - input.value(subCol1, ch) - input.value(subCol2, ch);
	outputValue += outputDifference;
	if (isRelevantForOutput(inputCol)) {
		output.value(outputCol, ch) = normalizer * outputValue;
		++outputCol;
	}
}

bool TriangularConvolutionFilter::isRelevantForOutput(int inputIndex) const {
	return noDownScaling || (inputIndex - downScaleOffset) % downScaleFactor == 0;
}

} /* namespace filtering */
} /* namespace imageprocessing */
