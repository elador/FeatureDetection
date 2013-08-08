/*
 * SpatialHistogramFilter.hpp
 *
 *  Created on: 30.05.2013
 *      Author: poschmann
 */

#ifndef SPATIALHISTOGRAMFILTER_HPP_
#define SPATIALHISTOGRAMFILTER_HPP_

#include "imageprocessing/HistogramFilter.hpp"
#include <vector>

using std::vector;

namespace imageprocessing {

/**
 * Filter that creates spatial histograms that may overlap. The input is an image with depth CV_8U containing bin information
 * (bin indices and optionally weights). It must have one, two or four channels. In case of one channel, it just contains the
 * bin index. The second channel adds a weight that will be divided by 255 to be between zero and one. The third channel is
 * another bin index and the fourth channel is the corresponding weight. The output will be a row vector of type CV_32F
 * containing the concatenated normalized histograms.
 */
class SpatialHistogramFilter : public HistogramFilter {
public:

	/**
	 * Constructs a new spatial histogram filter with square cells and blocks.
	 *
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellSize The preferred width and height of the cells in pixels (actual size might deviate).
	 * @param[in] blockSize The width and height of the blocks in cells.
	 * @param[in] interpolation Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] combinedHistograms Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
	 * @param[in] normalization The normalization method of the block histograms.
	 */
	SpatialHistogramFilter(unsigned int bins, int cellSize, int blockSize,
			bool interpolation, bool combineHistograms = true, Normalization normalization = Normalization::NONE);

	/**
	 * Constructs a new spatial histogram filter.
	 *
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellWidth The preferred width of the cells in pixels (actual width might deviate).
	 * @param[in] cellHeight The preferred height of the cells in pixels (actual height might deviate).
	 * @param[in] blockWidth The width of the blocks in cells.
	 * @param[in] blockHeight The height of the blocks in cells.
	 * @param[in] interpolation Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] combinedHistograms Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
	 * @param[in] normalization The normalization method of the block histograms.
	 */
	SpatialHistogramFilter(unsigned int bins, int cellWidth, int cellHeight, int blockWidth, int blockHeight,
			bool interpolation, bool combineHistograms = true, Normalization normalization = Normalization::NONE);

	~SpatialHistogramFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered) const;

private:

	/**
	 * Entry of the cache for the linear interpolation.
	 */
	struct CacheEntry {
		int index;    ///< First index (second one is index + 1).
		float weight; ///< Weight of the second index (weight of first index is 1 - weight).
	};

	/**
	 * Creates the cache contents if the size of the cache does not match the requested size.
	 *
	 * @param[in] cache The cache.
	 * @param[in] size The necessary size of the cache.
	 * @param[in] count The number of cells.
	 */
	void createCache(vector<CacheEntry>& cache, unsigned int size, int count) const;

	unsigned int bins; ///< The amount of bins inside the histogram.
	int cellWidth;     ///< The preferred width of the cells in pixels (actual width might deviate).
	int cellHeight;    ///< The preferred height of the cells in pixels (actual height might deviate).
	int blockWidth;    ///< The width of the blocks in cells.
	int blockHeight;   ///< The height of the blocks in cells.
	bool interpolation;     ///< Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	bool combineHistograms; ///< Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
	mutable vector<CacheEntry> rowCache; ///< Cache for the linear interpolation of the row indices.
	mutable vector<CacheEntry> colCache; ///< Cache for the linear interpolation of the column indices.
};

} /* namespace imageprocessing */
#endif /* SPATIALHISTOGRAMFILTER_HPP_ */
