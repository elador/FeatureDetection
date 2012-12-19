/*
 * VideoImageSink.cpp
 *
 *  Created on: 18.12.2012
 *      Author: poschmann
 */

#include "imageio/VideoImageSink.h"
#include <iostream>

using cv::Size;

namespace imageio {

VideoImageSink::VideoImageSink(const string filename, double fps, int fourcc) :
		filename(filename), fps(fps), fourcc(fourcc), writer() {}

VideoImageSink::~VideoImageSink() {}

void VideoImageSink::add(const Mat& image) {
	if (!writer.isOpened()) {
		if (!writer.open(filename, fourcc, fps, Size(image.cols, image.rows)))
			std::cerr << "Could not write video file '" << filename << "'" << std::endl;
	}
	writer << image;
}

} /* namespace imageio */
