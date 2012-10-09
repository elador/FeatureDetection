/*
 * VideoImageSource.cpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#include "imageio/VideoImageSource.h"
#include <iostream>

namespace imageio {

VideoImageSource::VideoImageSource(int device) : capture(device), frame() {
	if (!capture.isOpened())
		std::cerr << "Could not open stream from device " << device << std::endl;
}

VideoImageSource::VideoImageSource(std::string video) : capture(video), frame() {
	if (!capture.isOpened())
		std::cerr << "Could not open video file '" << video << "'" << std::endl;
}

VideoImageSource::~VideoImageSource() {
	capture.release();
}

const cv::Mat VideoImageSource::get() {
	capture >> frame;
	return frame;
}

} /* namespace imageio */
