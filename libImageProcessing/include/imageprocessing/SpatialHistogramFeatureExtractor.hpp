/*
 * SpatialHistogramFeatureExtractor.hpp
 *
 *  Created on: 30.05.2013
 *      Author: poschmann
 */

#ifndef SPATIALHISTOGRAMFEATUREEXTRACTOR_HPP_
#define SPATIALHISTOGRAMFEATUREEXTRACTOR_HPP_

#include "imageprocessing/HistogramFeatureExtractor.hpp"

namespace imageprocessing {

/**
 * Feature extractor that creates spatial histograms that may overlap. Builds upon a feature extractor that delivers
 * patches containing histogram information (bin indices and optionally weights).
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
	 * @param[in] combinedHistograms Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
	 * @param[in] normalization The normalization method of the block histograms.
	 */
	SpatialHistogramFeatureExtractor(shared_ptr<FeatureExtractor> extractor, unsigned int bins,
			int cellSize, int blockSize, bool combineHistograms = true, Normalization normalization = L2NORM);

	/**
	 * Constructs a new spatial histogram feature extractor.
	 *
	 * @param[in] extractor The underlying feature extractor (has to deliver histogram bin indices per pixel with optional weight).
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellWidth The preferred width of the cells in pixels (actual width might deviate).
	 * @param[in] cellHeight The preferred height of the cells in pixels (actual height might deviate).
	 * @param[in] blockWidth The width of the blocks in cells.
	 * @param[in] blockHeight The height of the blocks in cells.
	 * @param[in] combinedHistograms Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
	 * @param[in] normalization The normalization method of the block histograms.
	 */
	SpatialHistogramFeatureExtractor(shared_ptr<FeatureExtractor> extractor, unsigned int bins,
			int cellWidth, int cellHeight, int blockWidth, int blockHeight, bool combineHistograms = true, Normalization normalization = L2NORM);

	~SpatialHistogramFeatureExtractor();

	void update(const Mat& image);

	void update(shared_ptr<VersionedImage> image);

	shared_ptr<Patch> extract(int x, int y, int width, int height) const;

private:

	shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
	unsigned int bins; ///< The amount of bins inside the histogram.
	int cellWidth;     ///< The preferred width of the cells in pixels (actual width might deviate).
	int cellHeight;    ///< The preferred height of the cells in pixels (actual height might deviate).
	int blockWidth;    ///< The width of the blocks in cells.
	int blockHeight;   ///< The height of the blocks in cells.
	bool combineHistograms; ///< Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
};

} /* namespace imageprocessing */
#endif /* SPATIALHISTOGRAMFEATUREEXTRACTOR_HPP_ */
