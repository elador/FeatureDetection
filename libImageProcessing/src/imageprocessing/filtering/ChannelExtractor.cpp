/*
 * ChannelExtractor.cpp
 *
 *  Created on: 21.10.2015
 *      Author: poschmann
 */

#include "imageprocessing/filtering/ChannelExtractor.hpp"

using cv::Mat;
using std::vector;

namespace imageprocessing {
namespace filtering {

ChannelExtractor::ChannelExtractor(int channel) {
	createFromTo({channel});
}

ChannelExtractor::ChannelExtractor(const vector<int>& channels) {
	createFromTo(channels);
}

void ChannelExtractor::createFromTo(const vector<int>& channels) {
	fromTo.resize(2 * channels.size());
	for (int i = 0; i < channels.size(); ++i) {
		fromTo[2 * i] = channels[i]; // from
		fromTo[2 * i + 1] = i; // to
	}
}

Mat ChannelExtractor::applyTo(const Mat& image, Mat& filtered) const {
	filtered.create(image.rows, image.cols, CV_MAKE_TYPE(image.depth(), fromTo.size() / 2));
	cv::mixChannels({image}, {filtered}, fromTo);
	return filtered;
}

} /* namespace filtering */
} /* namespace imageprocessing */
