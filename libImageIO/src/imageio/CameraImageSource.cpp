/*
 * CameraImageSource.cpp
 *
 *  Created on: 27.09.2013
 *      Author: poschmann
 */

#include "imageio/CameraImageSource.hpp"
#include <iostream>
#include <string>

using boost::filesystem::path;
using cv::Mat;
using std::string;
using std::vector;
using std::invalid_argument;
using std::runtime_error;

namespace imageio {

CameraImageSource::CameraImageSource(int device) :
		ImageSource(std::to_string(device)), device(device), capture(device), frame(), frameCounter(0) {
	if (!capture.isOpened())
		throw invalid_argument("Could not open stream from device " + std::to_string(device));
}

CameraImageSource::~CameraImageSource() {
	capture.release();
}

void CameraImageSource::reset()
{
	capture.release();
	if (!capture.open(device))
		throw runtime_error("Could not open stream from device " + std::to_string(device));
	frameCounter = 0;
}

bool CameraImageSource::next()
{
	++frameCounter;	// We'll overflow after 2 years at 60fps... guess that's not a problem?
	return capture.read(frame);
}

const Mat CameraImageSource::getImage() const
{
	return frame;
}

path CameraImageSource::getName() const
{
	return path(std::to_string(frameCounter));
}

vector<path> CameraImageSource::getNames() const
{
	vector<path> tmp;
	tmp.push_back(path(std::to_string(frameCounter)));
	return tmp;
}


} /* namespace imageio */
