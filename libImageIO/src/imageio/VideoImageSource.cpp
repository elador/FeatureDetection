/*
 * VideoImageSource.cpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#include "imageio/VideoImageSource.hpp"
#include "boost/lexical_cast.hpp"
#include <iostream>
#include <string>

using boost::lexical_cast;
using std::string;

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

const Mat VideoImageSource::get() {
	capture >> frame;
	return frame;
}

const bool VideoImageSource::next()
{
	return true; // TODO: How can we find out here if we've reached the end of a video file?
}

const Mat VideoImageSource::getImage()
{
	return frame;	// TODO: What about the very first frame? We should initialize frame in the constructor.
					// TODO: This is currently flawed. A call to next() with a subsequent call to getImage()
					//       should return the next frame, not always the same!
}

const path VideoImageSource::getName()
{
	return path(lexical_cast<string>(frameCounter));
}

const vector<path> VideoImageSource::getNames()
{
	vector<path> tmp;
	tmp.push_back( path(lexical_cast<string>(frameCounter)));
	return tmp;
}


} /* namespace imageio */
