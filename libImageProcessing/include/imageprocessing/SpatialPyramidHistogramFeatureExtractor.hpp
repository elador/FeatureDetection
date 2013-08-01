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
 *
 * The patch data must be of type CV_8U and have one, two or four channels. In case of one channel, it just contains the
 * bin index. The second channel adds a weight that will be divided by 255 to be between zero and one. The third channel
 * is another bin index and the fourth channel is the corresponding weight.
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
			unsigned int bins, unsigned int level, Normalization normalization = Normalization::NONE);

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
