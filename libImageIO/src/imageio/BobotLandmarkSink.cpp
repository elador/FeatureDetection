/*
 * BobotLandmarkSink.cpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#include "imageio/BobotLandmarkSink.hpp"
#include "imageio/ImageSource.hpp"
#include "imageio/Landmark.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "opencv2/core/core.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Rect_;
using std::string;
using std::shared_ptr;
using std::runtime_error;

namespace imageio {

BobotLandmarkSink::BobotLandmarkSink(const string& videoFilename, int imageWidth, int imageHeight, const string& landmarkName) :
		videoFilename(videoFilename), imageWidth(imageWidth), imageHeight(imageHeight), imageSource(), landmarkName(landmarkName), output(), index(0) {
	output.setf(std::ios_base::fixed, std::ios_base::floatfield);
	output.precision(6);
}

BobotLandmarkSink::BobotLandmarkSink(const string& videoFilename, shared_ptr<ImageSource> imageSource, const string& landmarkName) :
		videoFilename(videoFilename), imageWidth(0), imageHeight(0), imageSource(imageSource), landmarkName(landmarkName), output(), index(0) {
	output.setf(std::ios_base::fixed, std::ios_base::floatfield);
	output.precision(6);
}

bool BobotLandmarkSink::isOpen() {
	return output.is_open();
}

void BobotLandmarkSink::open(const string& filename) {
	if (isOpen())
		throw runtime_error("BobotLandmarkSink: sink is already open");
	output.open(filename);
	output << videoFilename << '\n';
}

void BobotLandmarkSink::close() {
	output.close();
	index = 0;
}

void BobotLandmarkSink::add(const LandmarkCollection& collection) {
	if (!isOpen())
		throw runtime_error("BobotLandmarkSink: sink is not open");
	if (imageSource) {
		Mat image = imageSource->getImage();
		imageWidth = image.cols;
		imageHeight = image.rows;
	}
	const shared_ptr<Landmark> landmark = getLandmark(collection);
	if (landmark->isVisible()) {
		Rect_<float> rect = getLandmark(collection)->getRect();
		float x = rect.x / imageWidth;
		float y = rect.y / imageHeight;
		float width = rect.width / imageWidth;
		float height = rect.height / imageHeight;
		output << index++ << ' ' << x << ' ' << y << ' ' << width << ' ' << height << '\n';
	} else {
		output << index++ << " 0 0 0 0\n";
	}
}

const shared_ptr<Landmark> BobotLandmarkSink::getLandmark(const LandmarkCollection& collection) {
	if (landmarkName.empty())
		return collection.getLandmark();
	return collection.getLandmark(landmarkName);
}

} /* namespace imageio */
