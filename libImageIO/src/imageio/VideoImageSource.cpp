/*
 * VideoImageSource.cpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#include "imageio/VideoImageSource.hpp"
#include <iostream>
#include <string>
#include <stdexcept>

using cv::Mat;
using boost::filesystem::path;
using std::vector;
using std::string;
using std::invalid_argument;
using std::runtime_error;

namespace imageio {

VideoImageSource::VideoImageSource(string video) :
		ImageSource(video), video(video), capture(video), frame(), frameCounter(0) {
	if (!capture.isOpened())
		throw invalid_argument("Could not open video file '" + video + "'");
}

VideoImageSource::~VideoImageSource() {
	capture.release();
}

void VideoImageSource::reset()
{
	capture.release();
	if (!capture.open(video))
		throw runtime_error("Could not open video file '" + video + "'");
	frameCounter = 0;
}

bool VideoImageSource::next()
{
	++frameCounter;	// We'll overflow after 2 years at 60fps... guess that's not a problem?
	return capture.read(frame);
}

const Mat VideoImageSource::getImage() const
{
	return frame;
}

path VideoImageSource::getName() const
{
	return path(std::to_string(frameCounter));
}

vector<path> VideoImageSource::getNames() const
{
	vector<path> tmp;
	tmp.push_back(path(std::to_string(frameCounter)));
	return tmp;
}


} /* namespace imageio */
