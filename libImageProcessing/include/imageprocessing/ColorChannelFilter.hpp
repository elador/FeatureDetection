/*
 * ColorChannelFilter.hpp
 *
 *  Created on: 09.08.2013
 *      Author: poschmann
 */

#ifndef COLORCHANNELFILTER_HPP_
#define COLORCHANNELFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include <array>

namespace imageprocessing {

/**
 * Filter that takes a HSV image (type CV_8UC3) and produces a color histogram image. Each histogram bin is
 * represented by a channel of the resulting image, containing the contributing value of each pixel to that
 * bin. Optionally, there is an additional channel containing the color magnitude per pixel. The depth of
 * the resulting image is CV_8U.
 */
class ColorChannelFilter : public ImageFilter {
public:

	/**
	 * Constructs a new color channel filter.
	 *
	 * @param[in] bins The amount of bins.
	 * @param[in] magnitude Flag that indicates whether the result should have an additional magnitude channel.
	 */
	ColorChannelFilter(unsigned int bins, bool magnitude);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	/**
	 * Distribution of weight across two bins.
	 */
	struct BinData {
		int bin1, bin2;
		float weight1, weight2;
	};

	unsigned int bins; ///< The amount of bins.
	bool magnitude;    ///< Flag that indicates whether the result should have an additional magnitude channel.
	std::array<BinData, 181> color2bin; ///< The look-up table of bin data given a color value (hue).
};

} /* namespace imageprocessing */
#endif /* COLORCHANNELFILTER_HPP_ */
