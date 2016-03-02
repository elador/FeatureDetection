/*
 * GradientHistogramFilter.hpp
 *
 *  Created on: 11.12.2015
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_GRADIENTHISTOGRAMFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_GRADIENTHISTOGRAMFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/filtering/GradientMagnitudeFilter.hpp"
#include "imageprocessing/filtering/GradientOrientationFilter.hpp"
#include <array>
#include <memory>

namespace imageprocessing {
namespace filtering {

/**
 * Image filter that computes a gradient histogram based descriptor per pixel given gradients.
 *
 * The gradient image is expected to have a depth of CV_8U or CV_32F with an even amount of channels. Odd channels
 * are x gradients, whereas even channels are y gradients. For each pixel, the gradient with the strongest magnitude
 * is chosen.
 *
 * The resulting image contains a gradient orientation histogram per pixel. The histogram can span full gradients
 * in [0;2*pi), half gradients in [0;pi) or both concatenated. Additionally, there might be a magnitude channel, which
 * is the same as the sum of the histogram bin weights (of either the full or half histogram).
 */
class GradientHistogramFilter : public ImageFilter {
public:

	/**
	 * Creates a new gradient histogram filter that builds descriptors containing the signed gradient histogram (full
	 * gradients) and magnitude.
	 *
	 * @param[in] binCount Number of bins of the gradient histogram.
	 * @param[in] interpolate Flag that indicates whether to linearly interpolate between the neighboring bins.
	 * @param[in] normalizationRadius Radius of the magnitude normalization.
	 * @param[in] normalizationConstant Small normalization constant to prevent division by zero.
	 */
	static std::shared_ptr<GradientHistogramFilter> full(
			int binCount, bool interpolate, int normalizationRadius = 5, double normalizationConstant = 0.01);

	/**
	 * Creates a new gradient histogram filter that builds descriptors containing the unsigned gradient histogram (half
	 * gradients) and magnitude.
	 *
	 * @param[in] binCount Number of bins of the gradient histogram.
	 * @param[in] interpolate Flag that indicates whether to linearly interpolate between the neighboring bins.
	 * @param[in] normalizationRadius Radius of the magnitude normalization.
	 * @param[in] normalizationConstant Small normalization constant to prevent division by zero.
	 */
	static std::shared_ptr<GradientHistogramFilter> half(
			int binCount, bool interpolate, int normalizationRadius = 5, double normalizationConstant = 0.01);

	/**
	 * Creates a new gradient histogram filter that builds descriptors containing the signed gradient histogram (full
	 * gradients), unsigned gradient histogram (half gradients), and magnitude.
	 *
	 * @param[in] halfBinCount Number of bins of the unsigned gradient histogram (signed histogram will have twice that amount).
	 * @param[in] interpolate Flag that indicates whether to linearly interpolate between the neighboring bins.
	 * @param[in] normalizationRadius Radius of the magnitude normalization.
	 * @param[in] normalizationConstant Small normalization constant to prevent division by zero.
	 */
	static std::shared_ptr<GradientHistogramFilter> both(
			int halfBinCount, bool interpolate, int normalizationRadius = 5, double normalizationConstant = 0.01);

	/**
	 * Constructs a new gradient histogram filter.
	 *
	 * @param[in] binCount Number of bins of the gradient histogram.
	 * @param[in] interpolate Flag that indicates whether to linearly interpolate between the neighboring bins.
	 * @param[in] half Flag that indicates whether to integrate the unsigned gradient histogram into the descriptor.
	 * @param[in] full Flag that indicates whether to integrate the signed gradient histogram into the descriptor.
	 * @param[in] magnitude Flag that indicates whether to append the magnitude to the descriptor.
	 * @param[in] normalizationRadius Radius of the magnitude normalization.
	 * @param[in] normalizationConstant Small normalization constant to prevent division by zero.
	 */
	GradientHistogramFilter(int binCount, bool interpolate,
			bool half, bool full, bool magnitude, int normalizationRadius = 5, double normalizationConstant = 0.01);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	/**
	 * Draws a visualization of the unsigned histogram part of the feature descriptors. Each of the unsigned histogram
	 * bins will be visualized by a line along the virtual edge given by its orientation. The higher the bin value, the
	 * brighter the line.
	 *
	 * @param[in] descriptors Image containing feature descriptors with unsigned gradient histograms.
	 * @param[in] unsignedBinCount Number of bins of the unsigned gradient histogram.
	 * @param[in] offset Offset of the first bin channel inside the descriptors.
	 * @param[in] cellSize Size of a single descriptor visualization in pixels.
	 * @return Visualization of the unsigned histogram parts.
	 */
	static cv::Mat visualizeUnsignedHistograms(const cv::Mat& descriptors, int unsignedBinCount, int offset, int cellSize);

private:

	struct Bins {
		int bin1;
		int bin2;
		float weight1;
		float weight2;
	};

	struct LutEntry {
		Bins fullBins;
		Bins halfBins;
		float magnitude;
	};

	void createGradientLut();

	cv::Mat reduceToStrongestGradient(const cv::Mat& gradientImage) const;

	void computeGradientHistogramImage(const cv::Mat& singleGradientImage,
			const cv::Mat& magnitudeImage, cv::Mat& gradientHistogramImage) const;

	float computeOrientation(float gradientX, float gradientY) const;

	float computeMagnitude(float gradientX, float gradientY) const;

	float computeSquaredMagnitude(float gradientX, float gradientY) const;

	int computeBin(float value) const;

	// bin1 and bin2 are guaranteed to be different
	Bins computeInterpolatedBins(float value, float weight) const;

	int computeHalfBin(int bin) const;

	/**
	 * Draws square images with a line each, where the line visualizes the orientation of an unsigned histogram bin.
	 */
	static std::vector<cv::Mat> drawLines(int cellSize, int unsignedBinCount);

	/**
	 * Draws a visualization of the unsigned histogram part of feature descriptors into a float image.
	 */
	static cv::Mat drawFloatVisualization(const cv::Mat& descriptors, const std::vector<cv::Mat>& lines, int unsignedBinCount, int offset);

	/**
	 * Merges the line images by weighting them according to the unsigned histogram bin value and taking the
	 * maximum of each pixel value.
	 */
	static void mergeWeightedLines(cv::Mat& cell, const float* descriptor, const std::vector<cv::Mat>& lines, int unsignedBinCount, int offset);

	/**
	 * Converts the float visualization of the feature descriptors into a uchar visualization and rescales
	 * according to the highest weight. Takes into consideration negative weights.
	 */
	static cv::Mat rescaleToUchar(const cv::Mat& floatViz, const cv::Mat& descriptors, int unsignedBinCount, int offset);

	/**
	 * Determines the highest unsigned histogram bin value across all feature descriptors.
	 */
	static float getMaxWeight(const cv::Mat& descriptors, int unsignedBinCount, int offset);

	/**
	 * Determines the value of the given unsigned histogram bin.
	 */
	static float getWeight(const float* descriptor, int bin, int offset);

	GradientMagnitudeFilter magnitudeFilter; ///< Image filter that computes the gradient magnitudes given image gradients.
	GradientOrientationFilter orientationFilter; ///< Image filter that computes the gradient orientations given image gradients.
	bool fullAndHalf; ///< Flag that indicates whether to add full and half gradient histograms to the descriptor.
	bool magnitude; ///< Flag that indicates whether to append the magnitude to the descriptor.
	int binCount; ///< Number of bins of the gradient histogram.
	int halfBinCount; ///< Number of bins of the unsigned gradient histogram in case half and full gradient histograms are computed.
	int descriptorSize; ///< Number of channels of the resulting image.
	float value2bin; ///< Factor that computes the corresponding (floating point) bin when multiplied by a value.
	bool interpolate; ///< Flag that indicates whether to linearly interpolate between the neighboring bins.
	std::array<LutEntry, 256 * 256> binLut; ///< Look-up table for bin indices and weights of CV_8U gradient images.
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_GRADIENTHISTOGRAMFILTER_HPP_ */
