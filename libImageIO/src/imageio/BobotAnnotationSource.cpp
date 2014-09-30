/*
 * BobotAnnotationSource.cpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#include "imageio/BobotAnnotationSource.hpp"
#include "imageio/ImageSource.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

using cv::Rect;
using cv::Rect_;
using std::string;
using std::shared_ptr;
using std::invalid_argument;

namespace imageio {

BobotAnnotationSource::BobotAnnotationSource(const string& filename, int imageWidth, int imageHeight) :
		imageWidth(imageWidth), imageHeight(imageHeight), imageSource(), videoFilename(), positions(), index(-1) {
	readPositions(filename);
}

BobotAnnotationSource::BobotAnnotationSource(const string& filename, shared_ptr<ImageSource> imageSource) :
		imageWidth(0), imageHeight(0), imageSource(imageSource), videoFilename(), positions(), index(-1) {
	readPositions(filename);
}

void BobotAnnotationSource::readPositions(const string& filename) {
	string name;
	string line;
	std::ifstream file(filename.c_str());
	if (!file.is_open())
		throw invalid_argument("BobotAnnotationSource: file \"" + filename + "\" cannot be opened");
	if (file.good())
		std::getline(file, videoFilename);
	Rect_<float> position;
	while (file.good()) {
		if (!std::getline(file, line))
			break;
		// read values from line
		std::istringstream lineStream(line);
		lineStream >> name;
		lineStream >> position.x;
		lineStream >> position.y;
		lineStream >> position.width;
		lineStream >> position.height;
		positions.push_back(position);
	}
}

const string& BobotAnnotationSource::getVideoFilename() const {
	return videoFilename;
}

void BobotAnnotationSource::reset() {
	index = -1;
}

bool BobotAnnotationSource::next() {
	index++;
	return index < static_cast<int>(positions.size());
}

Rect BobotAnnotationSource::getAnnotation() const {
	if (index < 0 && index >= static_cast<int>(positions.size()))
		return Rect();
	if (imageSource) {
		const cv::Mat& image = imageSource->getImage();
		imageWidth = image.cols;
		imageHeight = image.rows;
	}
	return Rect(
			cvRound(positions[index].x * imageWidth),
			cvRound(positions[index].y * imageHeight),
			cvRound(positions[index].width * imageWidth),
			cvRound(positions[index].height * imageHeight));
}

} /* namespace imageio */
