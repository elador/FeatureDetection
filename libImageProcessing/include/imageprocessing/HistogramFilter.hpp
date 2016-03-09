/*
 * HistogramFilter.hpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#ifndef HISTOGRAMFILTER_HPP_
#define HISTOGRAMFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include <vector>

namespace imageprocessing {

/**
 * Filter whose results are histograms created over a regular grid of cells. Does only provide functions for creating and
 * normalizing histograms, can not be instantiated directly.
 */
class HistogramFilter : public ImageFilter {
public:

	/**
	 * Normalization method.
	 */
	enum class Normalization { NONE, L2NORM, L2HYS, L1NORM, L1SQRT };

	/**
	 * Constructs a new histogram feature extractor.
	 *
	 * @param[in] normalization The normalization method of the histograms.
	 */
	explicit HistogramFilter(Normalization normalization);

	virtual cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const = 0;

protected:

	/**
	 * Creates histograms over a regular grid of cells using an image containing bin information per pixel.
	 *
	 * The input image must have a depth CV_8U containing bin information (bin indices and optionally weights). It must have
	 * one, two or four channels. In case of one channel, it just contains the bin index. The second channel adds a weight
	 * that will be divided by 255 to be between zero and one. The third channel is another bin index and the fourth channel
	 * is the corresponding weight. The output is an image of size rowCount x columnCount and depth CV_32F, where each pixel
	 * contains one histogram with the bins each occupying one channel.
	 *
	 * @param[in] image Image containing bin data per pixel.
	 * @param[out] histograms Matrix containing a histogram in each cell.
	 * @param[in] binCount Bin count of the histogram.
	 * @param[in] rowCount Row count of the cell grid.
	 * @param[in] columnCount Column count of the cell grid.
	 * @param[in] interpolate Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 */
	void createCellHistograms(const cv::Mat& image, cv::Mat& histograms, int binCount, int rowCount, int columnCount, bool interpolate) const;

	/**
	 * Normalizes the given histogram.
	 *
	 * @param[in,out] histogram The histogram that should be normalized.
	 */
	void normalize(cv::Mat& histogram) const;

	static const float eps; ///< The small value being added to the norm to prevent division by zero.

	Normalization normalization; ///< The normalization method of the histograms.

private:

	/**
	 * Entry of the cache for the linear interpolation.
	 */
	struct CacheEntry {
		int index1;    ///< First index.
		int index2;    ///< Second index.
		float weight1; ///< Weight of the first index.
		float weight2; ///< Weight of the second index.
	};

	/**
	 * Creates the cache contents if the size of the cache does not match the requested size.
	 *
	 * @param[in] cache The cache.
	 * @param[in] size The necessary size of the cache.
	 * @param[in] count The number of cells.
	 */
	void createCache(std::vector<CacheEntry>& cache, unsigned int size, int count) const;

	/**
	 * Normalizes the given histogram according to L2-norm.
	 *
	 * @param[in,out] histogram The histogram that should be normalized.
	 */
	void normalizeL2(cv::Mat& histogram) const;

	/**
	 * Normalizes the given histogram according to L2-hys (L2-norm followed by clipping (limit maximum values to 0.2)
	 * and renormalizing).
	 *
	 * @param[in,out] histogram The histogram that should be normalized.
	 */
	void normalizeL2Hys(cv::Mat& histogram) const;

	/**
	 * Normalizes the given histogram according to L1-norm.
	 *
	 * @param[in,out] histogram The histogram that should be normalized.
	 */
	void normalizeL1(cv::Mat& histogram) const;

	/**
	 * Normalizes the given histogram according to L1-sqrt (L1-norm followed by computing the square root over all elements).
	 *
	 * @param[in,out] histogram The histogram that should be normalized.
	 */
	void normalizeL1Sqrt(cv::Mat& histogram) const;

	mutable std::vector<CacheEntry> rowCache; ///< Cache for the linear interpolation of the row indices.
	mutable std::vector<CacheEntry> colCache; ///< Cache for the linear interpolation of the column indices.
};

} /* namespace imageprocessing */
#endif /* HISTOGRAMFILTER_HPP_ */
