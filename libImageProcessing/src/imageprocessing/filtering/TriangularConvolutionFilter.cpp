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
		even(size % 2 == 0),
		size(size),
		radius((size - 1) / 2),
		addOffset1(radius + 2),
		addOffset2(radius),
		downScaleFactor(downScaling),
		downScaleOffset(downScaleFactor / 2),
		noDownScaling(downScaleFactor == 1),
		normalizer(alpha / ((even ? 4 : 1) * ((radius + 1) * (radius + 1) * (radius + 1) * (radius + 1)))),
		delta(delta / normalizer) {
	if (size < 1)
		throw invalid_argument("TriangularConvolutionFilter: size must be greater than zero, but was " + to_string(size));
	if (downScaling < 1)
		throw invalid_argument("TriangularConvolutionFilter: downScaling must be greater than zero, but was " + to_string(downScaling));
}

// the implementation is based on the observation that the difference of the difference between two subsequent filtered values
// of a one-dimensional triangular convolution differs only in three (or six) values, independent of the filter size
//
// consider for example the following one-dimensional values: a b c d e f g h
//
// with the following filter kernel of odd size: (1 2 3 2 1)/9
//
//   | filtered value       | 1st order difference | 2nd order difference
// a | a + 2a + 3a + 2b + c |(-a -a -a +a +b +c)   |(+a -2a +c)
// b | a + 2a + 3b + 2c + d | -a -a -a +b +c +d    |(+a -2a +d)
// c | a + 2b + 3c + 2d + e | -a -a -b +c +d +e    | +a -2b +e
// d | b + 2c + 3d + 2e + f | -a -b -c +d +e +f    | +a -2c +f
// e | c + 2d + 3e + 2f + g | -b -c -d +e +f +g    | +a -2d +g
// f | d + 2e + 3f + 2g + h | -c -d -e +f +g +h    | +b -2e +h
// g | e + 2f + 3g + 2h + h | -d -e -f +g +h +h    | +c -2f +h
// h | f + 2g + 3h + 2h + h | -e -f -g +h +h +h    | +d -2g +h
//
// or with the following filter kernel of even size: (1 3 5 5 3 1)/18
//
//   | filtered value            | 1st order difference  | 2nd order difference
// a | a + 3a + 5a + 5a + 3b + c |(-a -2a -2a +2a +2b +c)|
// b | a + 3a + 5a + 5b + 3c + d | -a -2a -2a +2b +2c +d |(+a +a -2a -2a +c +d)= (+a -2a +c) + (+a -2a +d)
// c | a + 3a + 5b + 5c + 3d + e | -a -2a -2a +2c +2d +e | +a +a -2a -2b +d +e = (+a -2a +d) + (+a -2b +e)
// d | a + 3b + 5c + 5d + 3e + f | -a -2a -2b +2d +2e +f | +a +a -2b -2c +e +f = (+a -2b +e) + (+a -2c +f)
// e | b + 3c + 5d + 5e + 3f + g | -a -2b -2c +2e +2f +g | +a +a -2c -2d +f +g = (+a -2c +f) + (+a -2d +g)
// f | c + 3d + 5e + 5f + 3g + h | -b -2c -2d +2f +2g +h | +a +b -2d -2e +g +h = (+a -2d +g) + (+b -2e +h)
// g | d + 3e + 5f + 5g + 3h + h | -c -2d -2e +2g +2h +h | +b +c -2e -2f +h +h = (+b -2e +h) + (+c -2f +h)
// h | e + 3f + 5g + 5h + 3h + h | -d -2e -2f +2g +2g +h | +c +d -2f -2g +h +h = (+c -2f +h) + (+d -2g +h)
//
// in both cases, the first order difference depends on the filter size, but the second order difference does not;
// it only depends on six (even size) or three (odd size) values;
// in the case of even sized kernel the second order difference of the odd kernel (with size - 1) is applied two times;
// this allows an implementation of the filtering whose processing time is approximately constant per pixel

Mat TriangularConvolutionFilter::applyTo(const Mat& image, Mat& filtered) const {
	if (image.depth() != CV_32F)
		throw invalid_argument("TriangularConvolutionFilter: image must have a depth of CV_32F, but was " + to_string(image.depth()));
	if (image.rows <= radius)
		throw invalid_argument("TriangularConvolutionFilter: image must have at least "
				+ to_string(radius + 1) + " rows, but had only " + to_string(image.rows));
	if (image.cols < addOffset1 + addOffset2)
		throw invalid_argument("TriangularConvolutionFilter: image must have at least "
				+ to_string(addOffset1 + addOffset2) + " columns, but had only " + to_string(image.cols));
	filtered.create(image.rows / downScaleFactor, image.cols / downScaleFactor, image.type());

	int channels = image.channels();
	int valuesPerRow = image.cols * channels;
	vector<float> difference(valuesPerRow);
	vector<float> intermediate(valuesPerRow);

	// initialize intermediate result (vertical filtering) of first row (for odd kernel)
	const float* firstRowValues = image.ptr<float>(0);
	for (int i = 0; i < valuesPerRow; ++i) {
		difference[i] = firstRowValues[i];
		intermediate[i] = difference[i];
	}
	for (int i = 0; i < radius; ++i) {
		const float* rowValues = image.ptr<float>(i + 1);
		for (int k = 0; k < valuesPerRow; ++k) {
			difference[k] += firstRowValues[k] + rowValues[k];
			intermediate[k] += difference[k];
		}
	}
	// initialize difference of intermediate result (for odd kernel)
	for (int i = 0; i < difference.size(); ++i)
		difference[i] = 0;
	for (int i = 0; i < radius; ++i) {
		const float* rowChannelValues = image.ptr<float>(i + 1);
		for (int k = 0; k < valuesPerRow; ++k) {
			difference[k] += rowChannelValues[k] - firstRowValues[k];
		}
	}
	// initialize intermediate result and its difference for even kernel
	vector<float> difference2(valuesPerRow); // second order difference (for odd kernel)
	if (even) {
		const float* rowValues = image.ptr<float>(addOffset2);
		for (int i = 0; i < valuesPerRow; ++i)
			difference2[i] = -firstRowValues[i] + rowValues[i];
		for (int i = 0; i < valuesPerRow; ++i) {
			// first output value of even kernel is the sum of the zeroth and first output value of the odd kernel
			float previousOutputValue = intermediate[i] - difference[i]; // zeroth output value of odd kernel
			intermediate[i] += previousOutputValue; // sum of zeroth and first output value of odd kernel
			// first output difference of even kernel is sum of zeroth and first output difference of odd kernel
			float previousOutputDifference = difference[i] - difference2[i]; // zeroth output difference of odd kernel
			difference[i] += previousOutputDifference; // sum of zeroth and first output difference of odd kernel
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
	for (int inputRow = 1; inputRow < inputRows; ++inputRow) {
		// compute intermediate result (vertical filtering) of row
		if (even) {
			for (int i = 0; i < valuesPerRow; ++i) {
				difference[i] += difference2[i];
			}
		}
		int addIndex1 = std::max(inputRow - addOffset1, 0);
		int subIndex  =          inputRow - 1;
		int addIndex2 = std::min(inputRow + addOffset2, lastInputRow);
		const float* addValues1 = image.ptr<float>(addIndex1);
		const float* subValues  = image.ptr<float>(subIndex);
		const float* addValues2 = image.ptr<float>(addIndex2);
		for (int i = 0; i < valuesPerRow; ++i) {
			difference2[i] = addValues1[i] - 2 * subValues[i] + addValues2[i];
			difference[i] += difference2[i];
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
	assert(cols >= addOffset1 + addOffset2);
	ConstRow input{inputValues, cols, channels};
	Row output{outputValues, cols / downScaleFactor, channels};

	for (int ch = 0; ch < input.channels; ++ch) {
		// initialize first output value (for odd kernel)
		float outputDifference = input.value(0, ch);
		float outputValue = delta + outputDifference;
		for (int col = 0; col < radius; ++col) {
			outputDifference += input.value(0, ch) + input.value(col + 1, ch);
			outputValue += outputDifference;
		}
		// initialize difference of output value (for odd kernel)
		outputDifference = 0;
		for (int col = 0; col < radius; ++col) {
			outputDifference += input.value(col + 1, ch) - input.value(0, ch);
		}
		// initialize first output and difference for even kernel
		if (even) {
			// first output value of even kernel is the sum of the zeroth and first output value of the odd kernel
			float previousOutputValue = outputValue - delta - outputDifference; // zeroth output value of odd kernel
			outputValue += previousOutputValue; // sum of zeroth and first output value of odd kernel
			// first output difference of even kernel is sum of zeroth and first output difference of odd kernel
			float odd2ndOrderDifference = -input.value(0, ch) + input.value(addOffset2, ch); // second order difference (for odd kernel)
			float previousOutputDifference = outputDifference - odd2ndOrderDifference; // zeroth output difference of odd kernel
			outputDifference += previousOutputDifference; // sum of zeroth and first output difference of odd kernel
			// compute output difference 1.5
			outputDifference += odd2ndOrderDifference; // the second even-kernel difference needs the first odd-kernel 2nd order difference
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
		for (int inputCol = 1; inputCol < addOffset1; ++inputCol) {
			int addCol1 = 0;
			int subCol = inputCol - 1;
			int addCol2 = inputCol + addOffset2;
			filterNextValue(input, output, ch, inputCol, outputCol,
					addCol1, subCol, addCol2, outputDifference, outputValue);
		}
		for (int inputCol = addOffset1; inputCol < std::min(inputCols, input.cols - addOffset2); ++inputCol) {
			int addCol1 = inputCol - addOffset1;
			int subCol = inputCol - 1;
			int addCol2 = inputCol + addOffset2;
			filterNextValue(input, output, ch, inputCol, outputCol,
					addCol1, subCol, addCol2, outputDifference, outputValue);
		}
		for (int inputCol = input.cols - addOffset2; inputCol < inputCols; ++inputCol) {
			int addCol1 = inputCol - addOffset1;
			int subCol = inputCol - 1;
			int addCol2 = lastInputCol;
			filterNextValue(input, output, ch, inputCol, outputCol,
					addCol1, subCol, addCol2, outputDifference, outputValue);
		}
	}
}

void TriangularConvolutionFilter::filterNextValue(ConstRow input, Row output, int ch, int inputCol, int& outputCol,
		int addCol1, int subCol, int addCol2, float& outputDifference, float& outputValue) const {
	float outputDifference2 = input.value(addCol1, ch) - 2 * input.value(subCol, ch) + input.value(addCol2, ch);
	outputDifference += outputDifference2;
	outputValue += outputDifference;
	if (even) // the next even-kernel difference needs the current odd-kernel 2nd order difference
		outputDifference += outputDifference2;
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
