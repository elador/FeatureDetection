/*
 * FhogFilter.hpp
 *
 *  Created on: 24.02.2016
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_FHOGFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_FHOGFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/filtering/FhogAggregationFilter.hpp"
#include "imageprocessing/filtering/GradientMagnitudeFilter.hpp"
#include "imageprocessing/filtering/GradientOrientationFilter.hpp"
#include <array>
#include <memory>

namespace imageprocessing {
namespace filtering {

/**
 * Image filter that computes FHOG descriptors over cells of pixels given image gradients. FHOG is
 * described in [1], implementation details were taken from [2].
 *
 * The gradient image is expected to have a depth of CV_8U or CV_32F with an even amount of channels. Odd
 * channels are x gradients, whereas even channels are y gradients. For each pixel, the gradient with the
 * strongest magnitude is chosen.
 *
 * The filtered image has a depth of CV_32F with as many channels as there are features, which are the
 * signed histogram bins, the unsigned histogram bins and four gradient energy channels. In case of 9
 * unsigned bins, the resulting descriptors have a length of 2*9 + 9 + 4 = 31.
 *
 * This filter is equivalent to chaining the following filters, but faster:
 *   imageprocessing::ConversionFilter (to float-image, normalize to 1 - without this conversion, the result differs slightly)
 *   imageprocessing::filtering::GradientFilter (with a size of 1)
 *   imageprocessing::filtering::GradientOrientationFilter (full orientations)
 *   imageprocessing::filtering::HistogramFilter (two times unsigned bin count, circular)
 *   imageprocessing::filtering::FhogAggregationFilter
 *
 * [1] Felzenszwalb et al., Object Detection with Discriminatively Trained Part-Based Models, PAMI, 2010.
 * [2] https://github.com/rbgirshick/voc-dpm/blob/master/features/features.cc
 */
class FhogFilter : public ImageFilter {
public:

	/**
	 * Constructs a new FHOG filter.
	 *
	 * @param[in] cellSize Size of the square cells in pixels.
	 * @param[in] unsignedBinCount Number of bins of the unsigned gradient histogram.
	 * @param[in] interpolateBins Flag that indicates whether to linearly interpolate between the neighboring bins.
	 * @param[in] interpolateCells Flag that indicates whether to bilinearly interpolate the pixel contributions to cells.
	 * @param[in] alpha Truncation threshold of the histogram bin values (applied after normalization).
	 */
	explicit FhogFilter(int cellSize = 8, int unsignedBinCount = 9,
			bool interpolateBins = false, bool interpolateCells = true, float alpha = 0.2f);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	/**
	 * Draws a visualization of the unsigned histogram part of the FHOG descriptors. Each of the unsigned
	 * histogram bins will be visualized by a line along the virtual edge given by its orientation. The
	 * higher the bin value, the brighter the line.
	 *
	 * @param[in] descriptors Image containing FHOG descriptors.
	 * @param[in] cellSize Size of a single descriptor visualization in pixels.
	 * @return Visualization of the unsigned histogram parts.
	 */
	static cv::Mat visualizeUnsignedHistograms(const cv::Mat& descriptors, int cellSize);

private:

	struct Coefficients {
		int index1;
		int index2;
		float weight1;
		float weight2;
	};

	struct LutEntry {
		Coefficients bins;
		float magnitude;
	};

	void createGradientLut();

	template<bool singleChannel>
	void computeSignedHistograms(const cv::Mat& singleGradientImage, cv::Mat& signedHistograms) const;

	std::vector<Coefficients> computeInterpolationCoefficents(int size, int count) const;

	template<bool singleChannel>
	Coefficients getBinCoefficients(const cv::Mat& image, int row, int col) const;

	void addToSignedHistograms(cv::Mat& signedHistograms, Coefficients rowCoeff, Coefficients colCoeff, Coefficients binCoeff) const;

	float computeOrientation(float gradientX, float gradientY) const;

	float computeMagnitude(float gradientX, float gradientY) const;

	int computeBin(float value) const;

	// bin1 and bin2 are guaranteed to be different
	Coefficients computeInterpolatedBins(float value, float weight) const;

	GradientMagnitudeFilter magnitudeFilter; ///< Image filter that computes the gradient magnitudes given image gradients.
	GradientOrientationFilter orientationFilter; ///< Image filter that computes the gradient orientations given image gradients.
	FhogAggregationFilter fhogAggregationFilter; ///< Image filter that computes the descriptor given signed gradient histograms.
	int cellSize; ///< Size of the square cells in pixels.
	int signedBinCount; ///< Number of bins of the signed gradient histogram.
	int unsignedBinCount; ///< Number of bins of the unsigned gradient histogram.
	float value2bin; ///< Factor that computes the corresponding (floating point) bin when multiplied by a value.
	bool interpolateBins; ///< Flag that indicates whether to linearly interpolate between the neighboring bins.
	bool interpolateCells; ///< Flag that indicates whether to bilinearly interpolate the pixel contributions to cells.
	std::array<LutEntry, 512 * 512> binLut; ///< Look-up table for bin indices and weights.
};

template<bool singleChannel>
inline void FhogFilter::computeSignedHistograms(const cv::Mat& image, cv::Mat& signedHistograms) const {
	assert(image.rows >= signedHistograms.rows * cellSize);
	assert(image.cols >= signedHistograms.cols * cellSize);
	std::vector<Coefficients> rowCoefficients = computeInterpolationCoefficents(signedHistograms.rows * cellSize, signedHistograms.rows);
	std::vector<Coefficients> colCoefficients = computeInterpolationCoefficents(signedHistograms.cols * cellSize, signedHistograms.cols);
	int channels = image.channels();
	for (int imageRow = 0; imageRow < rowCoefficients.size(); ++imageRow) {
		for (int imageCol = 0; imageCol < colCoefficients.size(); ++imageCol) {
			Coefficients binCoeff = getBinCoefficients<singleChannel>(image, imageRow, imageCol);
			addToSignedHistograms(signedHistograms, rowCoefficients[imageRow], colCoefficients[imageCol], binCoeff);
		}
	}
}

template<>
inline FhogFilter::Coefficients FhogFilter::getBinCoefficients<true>(const cv::Mat& image, int row, int col) const {
	int prevRow = std::max(row - 1, 0);
	int nextRow = std::min(row + 1, image.rows - 1);
	int prevCol = std::max(col - 1, 0);
	int nextCol = std::min(col + 1, image.cols - 1);
	int dx = image.at<uchar>(row, nextCol) - image.at<uchar>(row, prevCol) + 256;
	int dy = image.at<uchar>(nextRow, col) - image.at<uchar>(prevRow, col) + 256;
	return binLut[dy * 512 + dx].bins;
}

template<>
inline FhogFilter::Coefficients FhogFilter::getBinCoefficients<false>(const cv::Mat& image, int row, int col) const {
	int prevRow = std::max(row - 1, 0);
	int nextRow = std::min(row + 1, image.rows - 1);
	int prevCol = std::max(col - 1, 0);
	int nextCol = std::min(col + 1, image.cols - 1);
	const uchar* upValues = image.ptr<uchar>(prevRow, col);
	const uchar* downValues = image.ptr<uchar>(nextRow, col);
	const uchar* leftValues = image.ptr<uchar>(row, prevCol);
	const uchar* rightValues = image.ptr<uchar>(row, nextCol);
	int dx1 = rightValues[0] - leftValues[0] + 256;
	int dy1 = downValues[0] - upValues[0] + 256;
	int dx2 = rightValues[1] - leftValues[1] + 256;
	int dy2 = downValues[1] - upValues[1] + 256;
	int dx3 = rightValues[2] - leftValues[2] + 256;
	int dy3 = downValues[2] - upValues[2] + 256;
	int index1 = dy1 * 512 + dx1;
	int index2 = dy2 * 512 + dx2;
	int index3 = dy3 * 512 + dx3;
	if (binLut[index1].magnitude > binLut[index2].magnitude) {
		if (binLut[index1].magnitude > binLut[index3].magnitude)
			return binLut[index1].bins;
		else
			return binLut[index3].bins;
	} else {
		if (binLut[index2].magnitude > binLut[index3].magnitude)
			return binLut[index2].bins;
		else
			return binLut[index3].bins;
	}
}

inline void FhogFilter::addToSignedHistograms(cv::Mat& signedHistograms, Coefficients rowCoeff, Coefficients colCoeff, Coefficients binCoeff) const {
	if (interpolateCells) {
		float* histogram11 = signedHistograms.ptr<float>(rowCoeff.index1, colCoeff.index1);
		float* histogram12 = signedHistograms.ptr<float>(rowCoeff.index1, colCoeff.index2);
		float* histogram21 = signedHistograms.ptr<float>(rowCoeff.index2, colCoeff.index1);
		float* histogram22 = signedHistograms.ptr<float>(rowCoeff.index2, colCoeff.index2);
		if (interpolateBins) {
			histogram11[binCoeff.index1] += binCoeff.weight1 * rowCoeff.weight1 * colCoeff.weight1;
			histogram11[binCoeff.index2] += binCoeff.weight2 * rowCoeff.weight1 * colCoeff.weight1;
			histogram12[binCoeff.index1] += binCoeff.weight1 * rowCoeff.weight1 * colCoeff.weight2;
			histogram12[binCoeff.index2] += binCoeff.weight2 * rowCoeff.weight1 * colCoeff.weight2;
			histogram21[binCoeff.index1] += binCoeff.weight1 * rowCoeff.weight2 * colCoeff.weight1;
			histogram21[binCoeff.index2] += binCoeff.weight2 * rowCoeff.weight2 * colCoeff.weight1;
			histogram22[binCoeff.index1] += binCoeff.weight1 * rowCoeff.weight2 * colCoeff.weight2;
			histogram22[binCoeff.index2] += binCoeff.weight2 * rowCoeff.weight2 * colCoeff.weight2;
		} else {
			histogram11[binCoeff.index1] += binCoeff.weight1 * rowCoeff.weight1 * colCoeff.weight1;
			histogram12[binCoeff.index1] += binCoeff.weight1 * rowCoeff.weight1 * colCoeff.weight2;
			histogram21[binCoeff.index1] += binCoeff.weight1 * rowCoeff.weight2 * colCoeff.weight1;
			histogram22[binCoeff.index1] += binCoeff.weight1 * rowCoeff.weight2 * colCoeff.weight2;
		}
	} else {
		float* histogram = signedHistograms.ptr<float>(rowCoeff.index1, colCoeff.index1);
		if (interpolateBins) {
			histogram[binCoeff.index1] += binCoeff.weight1;
			histogram[binCoeff.index2] += binCoeff.weight2;
		} else {
			histogram[binCoeff.index1] += binCoeff.weight1;
		}
	}
}

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_FHOGFILTER_HPP_ */
