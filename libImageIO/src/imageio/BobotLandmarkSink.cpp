/*
 * BobotLandmarkSink.cpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#include "imageio/BobotLandmarkSink.hpp"
#include "imageio/Landmark.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "opencv2/core/core.hpp"

using cv::Rect_;

namespace imageio {

BobotLandmarkSink::BobotLandmarkSink(const string& landmarkName) :
		landmarkName(landmarkName), output(), index(0), imageWidth(0), imageHeight(0) {
	output.precision(6);
}

BobotLandmarkSink::~BobotLandmarkSink() {}

bool BobotLandmarkSink::isOpen() {
	return output.is_open();
}

void BobotLandmarkSink::open(const string& filename, const string& videoFilename, float imageWidth, float imageHeight) {
	output.open(filename);
	output << videoFilename << '\n';
	this->imageWidth = imageWidth;
	this->imageHeight = imageHeight;
}

void BobotLandmarkSink::close() {
	output.close();
}

void BobotLandmarkSink::add(const LandmarkCollection& collection) {
	Rect_<float> rect = getLandmark(collection).getRect();
	float x = rect.x / imageWidth;
	float y = rect.y / imageHeight;
	float width = rect.width / imageWidth;
	float height = rect.height / imageHeight;
	output << index++ << ' ' << x << ' ' << y << ' ' << width << ' ' << height << '\n';
}

const Landmark& BobotLandmarkSink::getLandmark(const LandmarkCollection& collection) {
	if (landmarkName.empty())
		return collection.getLandmark();
	return collection.getLandmark(landmarkName);
}

} /* namespace imageio */
