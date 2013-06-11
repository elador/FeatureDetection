/*
 * SpatialPyramidHistogramFeatureExtractor.hpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#ifndef SPATIALPYRAMIDHISTOGRAMFEATUREEXTRACTOR_HPP_
#define SPATIALPYRAMIDHISTOGRAMFEATUREEXTRACTOR_HPP_

#include "imageprocessing/HistogramFeatureExtractor.hpp"

namespace imageprocessing {

/**
 * Feature extractor that creates histograms over regions that get halfed in size at each level. The first level has one
 * region expaning over the whole patch, the second level has four regions, the third 16 and so on. The histograms will
 * be concatenated to result in the final feature descriptor.
 */
class SpatialPyramidHistogramFeatureExtractor : public HistogramFeatureExtractor {
public:

	/**
	 * Constructs a new spatial pyramid histogram feature extractor.
	 *
	 * @param[in] extractor The underlying feature extractor (has to deliver histogram bin indices per pixel with optional weight).
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] level The amount of levels of the pyramid.
	 * @param[in] normalization The normalization method of the histograms.
	 */
	SpatialPyramidHistogramFeatureExtractor(shared_ptr<FeatureExtractor> extractor,
			unsigned int bins, unsigned int level, Normalization normalization = NONE);

	~SpatialPyramidHistogramFeatureExtractor();

	void update(const Mat& image);

	void update(shared_ptr<VersionedImage> image);

	shared_ptr<Patch> extract(int x, int y, int width, int height) const;

private:

	shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
	unsigned int bins;  ///< The amount of bins inside the histogram.
	unsigned int level; ///< The amount of levels of the pyramid.
};

} /* namespace imageprocessing */
#endif /* SPATIALPYRAMIDHISTOGRAMFEATUREEXTRACTOR_HPP_ */
