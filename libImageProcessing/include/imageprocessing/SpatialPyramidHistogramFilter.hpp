/*
 * SpatialPyramidHistogramFilter.hpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#ifndef SPATIALPYRAMIDHISTOGRAMFILTER_HPP_
#define SPATIALPYRAMIDHISTOGRAMFILTER_HPP_

#include "imageprocessing/HistogramFilter.hpp"

namespace imageprocessing {

/**
 * Filter that creates histograms over regions that get halfed in size at each level. The first level has one region expanding
 * over the whole image, the second level has four regions, the third 16 and so on. The result will be a concatenation of all
 * histograms.
 *
 * The input is an image with depth CV_8U containing bin information (bin indices and optionally weights). It must have one,
 * two or four channels. In case of one channel, it just contains the bin index. The second channel adds a weight that will be
 * divided by 255 to be between zero and one. The third channel is another bin index and the fourth channel is the
 * corresponding weight. The output will be a row vector of type CV_32F containing the concatenated normalized histograms.
 */
class SpatialPyramidHistogramFilter : public HistogramFilter {
public:

	/**
	 * Constructs a new spatial pyramid histogram filter.
	 *
	 * @param[in] binCount The amount of bins inside the histogram.
	 * @param[in] levelCount The amount of levels of the pyramid.
	 * @param[in] interpolate Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] normalization The normalization method of the histograms.
	 */
	SpatialPyramidHistogramFilter(int binCount, int levelCount, bool interpolate = false, Normalization normalization = Normalization::NONE);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

protected:

	using HistogramFilter::createCellHistograms;

private:

	/**
	 * Creates the non-normalized histograms of a pyramid level using an image containing bin information per pixel.
	 *
	 * @param[in] image Image containing bin data per pixel.
	 * @param[out] histogramsValues Concatenated values of the level's histograms in row-major order.
	 * @param[in] level Level index where zero is the up-most level with only one histogram.
	 * @param[in] binCount Bin count of the histogram.
	 * @param[in] interpolate Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 */
	void createCellHistograms(const cv::Mat& image, float* histogramsValues, int level, int binCount, bool interpolate) const;

	/**
	 * Creates the histograms of a pyramid level by combining a block of 2x2 histograms of the previous level to one histogram.
	 *
	 * @param[in] cellHistogramsValues Concatenated values of the previous level's histograms in row-major order.
	 * @param[out] blockHistogramsValues Concatenated values of this level's histograms in row-major order.
	 * @param[in] level Level index where zero is the up-most level with only one histogram.
	 * @param[in] binCount Bin count of the histograms.
	 */
	void combineHistograms(const float* cellHistogramsValues, float* blockHistogramsValues, int level, int binCount) const;

	/**
	 * Normalizes histograms whose values are concatenated into one vector.
	 *
	 * @param[in,out] histogramsValues Concatenated values of the histograms.
	 * @param[in] histogramCount Histogram count.
	 * @param[in] binCount Bin count of the histograms.
	 */
	void normalizeHistograms(float* histogramsValues, int histogramCount, int binCount) const;

	int binCount;       ///< The amount of bins inside the histogram.
	int maxLevel;       ///< The index of the lowest pyramid level.
	int histogramCount; ///< The total amount of histograms.
	bool interpolate;   ///< Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
};

} /* namespace imageprocessing */
#endif /* SPATIALPYRAMIDHISTOGRAMFILTER_HPP_ */
