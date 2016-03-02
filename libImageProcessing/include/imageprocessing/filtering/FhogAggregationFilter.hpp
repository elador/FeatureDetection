/*
 * FhogAggregationFilter.hpp
 *
 *  Created on: 17.02.2016
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_FHOGAGGREGATIONFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_FHOGAGGREGATIONFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/filtering/AggregationFilter.hpp"
#include <array>

namespace imageprocessing {
namespace filtering {

/**
 * Image filter that computes FHOG descriptors over cells of pixels given signed gradient histograms.
 * FHOG is described in [1], implementation details were taken from [2].
 *
 * Expects an image of depth CV_32F with as many channels as there are bins inside the signed gradient
 * histogram. This filter then adds the unsigned gradient histogram (which has half as many bins) and
 * normalizes the histograms using the gradient magnitude (aka gradient energy) and caps values above a
 * pre-defined threshold.
 *
 * The filtered image has a depth of CV_32F with as many channels as there are features, which are the
 * signed histogram bins, the unsigned histogram bins and four gradient energy channels. In case of 18
 * signed bins, the resulting descriptors have a length of 18 + 18/2 + 4 = 31.
 *
 * [1] Felzenszwalb et al., Object Detection with Discriminatively Trained Part-Based Models, PAMI, 2010.
 * [2] https://github.com/rbgirshick/voc-dpm/blob/master/features/features.cc
 */
class FhogAggregationFilter : public ImageFilter {
public:

	/**
	 * Constructs a new FHOG aggregation filter.
	 *
	 * @param[in] cellSize Size of the square cells in pixels.
	 * @param[in] interpolate Flag that indicates whether to bilinearly interpolate the pixel contributions to cells.
	 * @param[in] alpha Truncation threshold of the histogram bin values (applied after normalization).
	 */
	explicit FhogAggregationFilter(int cellSize, bool interpolate = true, float alpha = 0.2f);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& descriptors) const;

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

	/**
	 * Computes the FHOG descriptors.
	 *
	 * @param[out] descriptors FHOG descriptors.
	 * @param[in] histograms Signed gradient histograms. Can be same as descriptors if it has the correct channel count.
	 * @param[in] signedBinCount Number of bins of the signed gradient histogram.
	 */
	void computeDescriptors(cv::Mat& descriptors, const cv::Mat& histograms, int signedBinCount) const;

	/**
	 * Computes the squared gradient energies for each gradient histogram.
	 *
	 * @param[in] histograms Signed gradient histograms.
	 * @param[in] signedBinCount Number of bins of the signed gradient histogram.
	 * @return Squared gradient energies.
	 */
	cv::Mat computeGradientEnergies(const cv::Mat& histograms, int signedBinCount) const;

	/**
	 * Computes the squared gradient energy for a gradient histogram.
	 * @param[in] signedHistogram Values of the signed gradient histogram.
	 * @param[in] unsignedBinCount Bin count of the unsigned gradient histogram (signed histogram has twice the amount).
	 * @return Squared gradient energy.
	 */
	float computeGradientEnergy(const float* signedHistogram, int unsignedBinCount) const;

	/**
	 * Computes the FHOG descriptors.
	 *
	 * @param[out] descriptors FHOG descriptors.
	 * @param[in] histograms Signed gradient histograms.
	 * @param[in] energies Squared gradient energies.
	 * @param[in] signedBinCount Number of bins of the signed gradient histogram.
	 */
	void computeDescriptors(cv::Mat& descriptors, const cv::Mat& histograms, const cv::Mat& energies, int signedBinCount) const;

	/**
	 * Computes the four normalizers for a histogram, that are computed from the squared gradient energy
	 * of four combined histograms.
	 *
	 * @param[in] energies Squared gradient energies (for each histogram).
	 * @param[in] row Row index of the histogram to compute the normalizers of.
	 * @param[in] col Column index of the histogram to compute the normalizers of.
	 * @return The four different normalizers of the histogram.
	 */
	std::array<float, 4> computeNormalizers(const cv::Mat& energies, int row, int col) const;

	/**
	 * Computes a descriptor.
	 *
	 * @param[out] descriptor Descriptor.
	 * @param[in] signedHistogram Signed gradient histogram.
	 * @param[in] normalizers Normalizers of the histogram.
	 * @param[in] signedBinCount Bin count of the signed gradient histogram.
	 * @param[in] unsignedBinCount Bin count of the unsigned gradient histogram.
	 */
	void computeDescriptor(float* descriptor, const float* signedHistogram,
			const std::array<float, 4>& normalizers, int signedBinCount, int unsignedBinCount) const;

	/**
	 * Normalizes a value with four different normalizers.
	 */
	std::array<float, 4> computeNormalizedValues(float value, const std::array<float, 4>& normalizers) const;

	/**
	 * Computes the sum of four values.
	 */
	float computeSum(const std::array<float, 4>& values) const;

	/**
	 * Adds four values to four sums.
	 */
	void addTo(std::array<float, 4>& sums, const std::array<float, 4>& values) const;

	static const float eps; ///< The small value being added to the norm to prevent division by zero.
	float alpha; ///< Truncation threshold of the histogram bin values (applied after normalization).
	AggregationFilter aggregationFilter; ///< Filter that aggregates the histogram values over the cells.

	friend class FhogFilter;
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_FHOGAGGREGATIONFILTER_HPP_ */
