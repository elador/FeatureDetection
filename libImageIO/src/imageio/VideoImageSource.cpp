/*
 * VideoImageSource.cpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#include "imageio/VideoImageSource.hpp"
#include <stdexcept>

using cv::Mat;
using std::vector;
using std::string;
using std::invalid_argument;
using std::runtime_error;

namespace imageio {

VideoImageSource::VideoImageSource(string video) : video(video), capture(video), frame() {
	if (!capture.isOpened())
		throw invalid_argument("VideoImageSource: Could not open video file '" + video + "'");
}

VideoImageSource::~VideoImageSource() {
	capture.release();
}

void VideoImageSource::reset() {
	capture.release();
	if (!capture.open(video))
		throw runtime_error("VideoImageSource: Could not open video file '" + video + "'");
}

bool VideoImageSource::next() {
	return capture.read(frame);
}

const Mat VideoImageSource::getImage() const {
	return frame;
}

} /* namespace imageio */
