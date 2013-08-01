/*
 * SpatialHistogramFeatureExtractor.hpp
 *
 *  Created on: 30.05.2013
 *      Author: poschmann
 */

#ifndef SPATIALHISTOGRAMFEATUREEXTRACTOR_HPP_
#define SPATIALHISTOGRAMFEATUREEXTRACTOR_HPP_

#include "imageprocessing/HistogramFeatureExtractor.hpp"
#include <vector>

using std::vector;

namespace imageprocessing {

/**
 * Feature extractor that creates spatial histograms that may overlap. Builds upon a feature extractor that delivers
 * patches containing histogram information (bin indices and optionally weights).
 *
 * The patch data must be of type CV_8U and have one, two or four channels. In case of one channel, it just contains the
 * bin index. The second channel adds a weight that will be divided by 255 to be between zero and one. The third channel
 * is another bin index and the fourth channel is the corresponding weight.
 */
class SpatialHistogramFeatureExtractor : public HistogramFeatureExtractor {
public:

	/**
	 * Constructs a new spatial histogram feature extractor with square cells and blocks.
	 *
	 * @param[in] extractor The underlying feature extractor (has to deliver histogram bin indices per pixel with optional weight).
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellSize The preferred width and height of the cells in pixels (actual size might deviate).
	 * @param[in] blockSize The width and height of the blocks in cells.
	 * @param[in] interpolation Flat that indicates whether the bin value of a pixel should be divided between neighboring cells.
	 * @param[in] combinedHistograms Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
	 * @param[in] normalization The normalization method of the block histograms.
	 */
	SpatialHistogramFeatureExtractor(shared_ptr<FeatureExtractor> extractor, unsigned int bins,
			int cellSize, int blockSize, bool interpolation,
			bool combineHistograms = true, Normalization normalization = Normalization::NONE);

	/**
	 * Constructs a new spatial histogram feature extractor.
	 *
	 * @param[in] extractor The underlying feature extractor (has to deliver histogram bin indices per pixel with optional weight).
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellWidth The preferred width of the cells in pixels (actual width might deviate).
	 * @param[in] cellHeight The preferred height of the cells in pixels (actual height might deviate).
	 * @param[in] blockWidth The width of the blocks in cells.
	 * @param[in] blockHeight The height of the blocks in cells.
	 * @param[in] interpolation Flat that indicates whether the bin value of a pixel should be divided between neighboring cells.
	 * @param[in] combinedHistograms Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
	 * @param[in] normalization The normalization method of the block histograms.
	 */
	SpatialHistogramFeatureExtractor(shared_ptr<FeatureExtractor> extractor, unsigned int bins,
			int cellWidth, int cellHeight, int blockWidth, int blockHeight, bool interpolation,
			bool combineHistograms = true, Normalization normalization = Normalization::NONE);

	~SpatialHistogramFeatureExtractor();

	void update(const Mat& image);

	void update(shared_ptr<VersionedImage> image);

	shared_ptr<Patch> extract(int x, int y, int width, int height) const;

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
	 * @param[in] cellSize The size (rows or columns) of the cell.
	 */
	void createCache(vector<CacheEntry>& cache, unsigned int size, int cellSize) const;

	shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
	unsigned int bins; ///< The amount of bins inside the histogram.
	int cellWidth;     ///< The preferred width of the cells in pixels (actual width might deviate).
	int cellHeight;    ///< The preferred height of the cells in pixels (actual height might deviate).
	int blockWidth;    ///< The width of the blocks in cells.
	int blockHeight;   ///< The height of the blocks in cells.
	bool interpolation;     ///< Flat that indicates whether the bin value of a pixel should be divided between neighboring cells.
	bool combineHistograms; ///< Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
	mutable vector<CacheEntry> rowCache; ///< Cache for the linear interpolation of the row indices.
	mutable vector<CacheEntry> colCache; ///< Cache for the linear interpolation of the column indices.
};

} /* namespace imageprocessing */
#endif /* SPATIALHISTOGRAMFEATUREEXTRACTOR_HPP_ */
