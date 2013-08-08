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
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] level The amount of levels of the pyramid.
	 * @param[in] normalization The normalization method of the histograms.
	 */
	SpatialPyramidHistogramFilter(unsigned int bins, unsigned int level, Normalization normalization = Normalization::NONE);

	~SpatialPyramidHistogramFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered) const;

private:

	unsigned int bins;  ///< The amount of bins inside the histogram.
	unsigned int level; ///< The amount of levels of the pyramid.
};

} /* namespace imageprocessing */
#endif /* SPATIALPYRAMIDHISTOGRAMFILTER_HPP_ */
