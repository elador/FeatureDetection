/*
 * BoxConvolutionFilter.hpp
 *
 *  Created on: 26.01.2016
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_BOXCONVOLUTIONFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_BOXCONVOLUTIONFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {
namespace filtering {

/**
 * Image filter that convolves the image using a 2D box (rectangular) kernel.
 *
 * The convolution is performed in constant time per pixel, independent of the kernel size. The pixels just outside
 * the image behave according to OpenCVs BORDER_REPLICATE, meaning the border values are replicated outside the image
 * boundaries. Additionally, the convolved image may be downsampled by skipping rows and columns.
 */
class BoxConvolutionFilter : public ImageFilter {
public:

	/**
	 * Constructs a new box convolution filter.
	 *
	 * @param[in] size Size of the one-dimensional filter (two-dimensional filter is size x size).
	 * @param[in] downScaling Optional factor by which the filtered image should be smaller than the input image.
	 * @param[in] alpha Optional factor by which to scale the filtered values.
	 * @param[in] delta Optional offset for the filtered values.
	 */
	explicit BoxConvolutionFilter(int size, int downScaling = 1, float alpha = 1, float delta = 0);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	/**
	 * @return Size of the one-dimensional filter (two-dimensional filter is size x size).
	 */
	int getSize() const {
		return size;
	}

private:

	struct Row {
		float* values;
		const int cols;
		const int channels;
		float& value(int col, int channel) {
			return values[col * channels + channel];
		}
	};

	struct ConstRow {
		const float* values;
		const int cols;
		const int channels;
		float value(int col, int channel) const {
			return values[col * channels + channel];
		}
	};

	/**
	 * Applies a one-dimensional box filter to a row of values.
	 *
	 * @param[in] inputValues Row of values (of size cols * channels).
	 * @param[out] outputValues Row of filtered values (of size (cols / downScaleFactor) * channels).
	 * @param[in] cols Number of columns.
	 * @param[in] channels Number of channels.
	 */
	void filterRow(const float* inputValues, float* outputValues, int cols, int channels) const;

	/**
	 * Computes the next filtered value by adding to and subtracting from the current output value.
	 *
	 * @param[in] input Row of input values.
	 * @param[out] output Row of output values.
	 * @param[in] ch Channel index.
	 * @param[in] inputCol Column index of the input value that is about to be filtered.
	 * @param[in,out] outputCol Column index of the output where the filtered value is stored. If stored, this index will be incremented.
	 * @param[in] addCol Column index of the input value that is added to the current output value.
	 * @param[in] subCol Column index of the input value that is subtracted from the current output value.
	 * @param[in,out] outputValue Current filtered value. Will store the computed filtered value.
	 */
	void filterNextValue(ConstRow input, Row output, int ch, int inputCol, int& outputCol,
			int addCol, int subCol, float& outputValue) const;

	bool shouldWriteToOutput(int index) const;

	bool odd; ///< Flag that indicates whether the filter size is odd.
	int size; ///< Size of the one-dimensional filter (two-dimensional filter is size x size).
	int radius; ///< Half filter size (rounded down if odd).
	int downScaleFactor; ///< Factor by which the filtered image should be smaller than the input image.
	int downScaleOffset; ///< Offset of the rows and columns that should be kept when downscaling.
	bool noDownScaling; ///< Flag that indicates whether the filtered image has the same size as the input image.
	float normalizer; ///< Normalizer of the two-dimensional box convolution kernel.
	float delta; ///< Offset for the filtered values.
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_BOXCONVOLUTIONFILTER_HPP_ */
