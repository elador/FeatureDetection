/*
 * PyramidHogFilter.hpp
 *
 *  Created on: 05.09.2013
 *      Author: poschmann
 */

#ifndef PYRAMIDHOGFILTER_HPP_
#define PYRAMIDHOGFILTER_HPP_

#include "imageprocessing/HistogramFilter.hpp"

namespace imageprocessing {

/**
 * Filter that creates a spatial pyramid of histograms of oriented gradients (HOG) based on images containing gradient
 * bin information. The normalization method is L2NORM and cannot be changed.
 *
 * The input is an image with depth CV_8U containing bin information (bin indices and optionally weights). It must have
 * one, two or four channels. In case of one channel, it just contains the bin index. The second channel adds a weight
 * that will be divided by 255 to be between zero and one. The third channel is another bin index and the fourth channel
 * is the corresponding weight. The output will be a row vector of type CV_32F.
 */
class PyramidHogFilter : public HistogramFilter {
public:

	/**
	 * Constructs a new pyramid HOG filter.
	 *
	 * @param[in] binCount The amount of bins inside the histogram.
	 * @param[in] levelCount The amount of levels of the pyramid.
	 * @param[in] interpolate Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 */
	PyramidHogFilter(int binCount, int levelCount, bool interpolate = false, bool signedAndUnsigned = false);

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
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 */
	void createCellHistograms(const cv::Mat& image, float* histogramsValues, int level, int binCount, bool interpolate, bool signedAndUnsigned) const;

	/**
	 * Copies the cell histograms from source to destination. If both signed and unsigned gradients should be combined
	 * within the histograms, then the destination histograms will be 1.5 times bigger because of the added unsigned bins.
	 *
	 * @param[in] source Source of the concatenated values of the cell histograms in row-major order.
	 * @param[out] destination Destination of the concatenated values of the cell histograms in row-major order.
	 * @param[in] binCount Bin count of the histogram.
	 * @param[in] cellCount Cell/histogram count.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 */
	void copyCellHistograms(const float* source, float* destination, int binCount, int cellCount, bool signedAndUnsigned) const;

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
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 */
	void normalizeHistograms(float* histogramsValues, int histogramCount, int binCount, bool signedAndUnsigned) const;

	int binCount;       ///< The amount of bins inside the histogram.
	int maxLevel;       ///< The index of the lowest pyramid level.
	int histogramCount; ///< The total amount of histograms.
	bool interpolate;   ///< Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	bool signedAndUnsigned; ///< Flag that indicates whether signed and unsigned gradients should be used.
};

} /* namespace imageprocessing */
#endif /* PYRAMIDHOGFILTER_HPP_ */
