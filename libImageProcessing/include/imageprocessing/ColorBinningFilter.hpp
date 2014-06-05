/*
 * ColorBinningFilter.hpp
 *
 *  Created on: 09.08.2013
 *      Author: poschmann
 */

#ifndef COLORBINNINGFILTER_HPP_
#define COLORBINNINGFILTER_HPP_

#include "imageprocessing/BinningFilter.hpp"
#include <array>

namespace imageprocessing {

/**
 * Image filter that creates a four-channel image containing color bin information per pixel, where channels one
 * and three correspond to histogram bins (according to hue) and channels two and four are the corresponding
 * weights (according to saturation, value and the distance of the hue to the center of the bin). The input image
 * has to have a HSV color space (three channels of depth CV_8U).
 */
class ColorBinningFilter : public BinningFilter {
public:

	/**
	 * Constructs a new color binning filter.
	 *
	 * @param[in] bins The amount of bins.
	 */
	explicit ColorBinningFilter(unsigned int bins);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	unsigned int getBinCount() const;

private:

	/**
	 * Distribution of weight across two bins.
	 */
	struct BinData {
		int bin1, bin2;
		float weight1, weight2;
	};

	unsigned int bins; ///< The amount of bins.
	std::array<BinData, 181> color2bin; ///< The look-up table of bin data given a color value (hue).
};

} /* namespace imageprocessing */
#endif /* COLORBINNINGFILTER_HPP_ */
