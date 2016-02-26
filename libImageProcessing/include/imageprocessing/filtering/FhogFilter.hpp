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
	FhogFilter(int cellSize, int unsignedBinCount, bool interpolateBins = false, bool interpolateCells = true, float alpha = 0.2f);

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

	cv::Mat reduceToStrongestGradient(const cv::Mat& gradientImage) const;

	void computeSignedHistograms(const cv::Mat& singleGradientImage, cv::Mat& signedHistograms) const;

	std::vector<Coefficients> computeInterpolationCoefficents(int size, int count) const;

	float computeOrientation(float gradientX, float gradientY) const;

	float computeMagnitude(float gradientX, float gradientY) const;

	float computeSquaredMagnitude(float gradientX, float gradientY) const;

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
	std::array<LutEntry, 256 * 256> binLut; ///< Look-up table for bin indices and weights of CV_8U gradient images.
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_FHOGFILTER_HPP_ */
