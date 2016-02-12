/*
 * ChannelExtractor.hpp
 *
 *  Created on: 21.10.2015
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_CHANNELEXTRACTOR_HPP_
#define IMAGEPROCESSING_FILTERING_CHANNELEXTRACTOR_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include <vector>

namespace imageprocessing {
namespace filtering {

/**
 * Image filter that extracts channels from the original image.
 */
class ChannelExtractor : public ImageFilter {
public:

	/**
	 * Constructs a new channel selector that selects a single channel.
	 *
	 * @param[in] channel Channel that should be selected.
	 */
	explicit ChannelExtractor(int channel);

	/**
	 * Constructs a new channel selector.
	 *
	 * @param[in] channels Channels that should be extracted in that order.
	 */
	explicit ChannelExtractor(const std::vector<int>& channels);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	void createFromTo(const std::vector<int>& channels);

	std::vector<int> fromTo; ///< Channel index pairs specifying input (2k) and output (2k+1) channel indices.
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_CHANNELEXTRACTOR_HPP_ */
