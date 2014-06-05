/*
 * GradientChannelFilter.hpp
 *
 *  Created on: 05.08.2013
 *      Author: poschmann
 */

#ifndef GRADIENTCHANNELFILTER_HPP_
#define GRADIENTCHANNELFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include <array>

namespace imageprocessing {

/**
 * Filter that takes a gradient image and produces a gradient orientation histogram image. The image is expected
 * to be of type CV_8UC2, where the first channel corresponds to the gradient of x and the second channel
 * corresponds to the gradient of y. Each histogram bin is represented by a channel of the resulting image,
 * containing the contributing value of each pixel to that bin. Optionally, there is an additional channel
 * containing the gradient magnitude per pixel. The depth of the resulting image is CV_8U.
 */
class GradientChannelFilter : public ImageFilter {
public:

	/**
	 * Constructs a new gradient channel filter.
	 *
	 * @param[in] bins The amount of bins.
	 * @param[in] magnitude Flag that indicates whether the result should have an additional magnitude channel.
	 * @param[in] signedGradients Flag that indicates whether signed gradients (direction from 0° to 360°) should be used.
	 */
	GradientChannelFilter(unsigned int bins, bool magnitude, bool signedGradients = false);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	/**
	 * Distribution of weight across two bins.
	 */
	struct BinData {
		int bin1, bin2;
		uchar weight1, weight2;
	};

	unsigned int bins; ///< The amount of bins.
	bool magnitude;    ///< Flag that indicates whether the result should have an additional magnitude channel.
	std::array<BinData, 256 * 256> gradient2bin; ///< The look-up table of bin data given a gradient code.
};

} /* namespace imageprocessing */
#endif /* GRADIENTCHANNELFILTER_HPP_ */
