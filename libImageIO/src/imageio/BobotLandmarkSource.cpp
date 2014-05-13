/*
 * BobotLandmarkSource.cpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#include "imageio/BobotLandmarkSource.hpp"
#include "imageio/ImageSource.hpp"
#include "imageio/RectLandmark.hpp"
#include <fstream>
#include <sstream>
#include <utility>
#include <memory>
#include <stdexcept>

using cv::Rect_;
using boost::filesystem::path;
using std::string;
using std::vector;
using std::unordered_map;
using std::shared_ptr;
using std::make_pair;
using std::make_shared;
using std::invalid_argument;
using std::runtime_error;

namespace imageio {

const string BobotLandmarkSource::landmarkName = "target";

BobotLandmarkSource::BobotLandmarkSource(const string& filename, int imageWidth, int imageHeight) :
		imageWidth(imageWidth), imageHeight(imageHeight), imageSource(), videoFilename(), positions(), name2index(), index2name(), index(-1) {
	readPositions(filename);
}

BobotLandmarkSource::BobotLandmarkSource(const string& filename, shared_ptr<ImageSource> imageSource) :
		imageWidth(0), imageHeight(0), imageSource(imageSource), videoFilename(), positions(), name2index(), index2name(), index(-1) {
	readPositions(filename);
}

void BobotLandmarkSource::readPositions(const string& filename) {
	string name;
	string line;
	std::ifstream file(filename.c_str());
	if (!file.is_open())
		throw invalid_argument("BobotLandmarkSource: file \"" + filename + "\" cannot be opened");
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
		name2index.emplace(name, positions.size() - 1);
	}
}

const string& BobotLandmarkSource::getVideoFilename() const {
	return videoFilename;
}

void BobotLandmarkSource::reset() {
	index = -1;
}

bool BobotLandmarkSource::next() {
	index++;
	return index < static_cast<int>(positions.size());
}

LandmarkCollection BobotLandmarkSource::get(const path& imagePath) {
	const auto iterator = name2index.find(imagePath.string());
	if (iterator == name2index.end())
		index = -1;
	else
		index = static_cast<int>(iterator->second);
	return getLandmarks();
}

LandmarkCollection BobotLandmarkSource::getLandmarks() const {
	LandmarkCollection collection;
	if (index >= 0 && index < static_cast<int>(positions.size())) {
		if (imageSource) {
			const cv::Mat& image = imageSource->getImage();
			imageWidth = image.cols;
			imageHeight = image.rows;
		}
		const Rect_<float>& relativePosition = positions[index];
		if (relativePosition.x == 0 && relativePosition.y == 0 && relativePosition.width == 0 && relativePosition.height == 0) {
			collection.insert(make_shared<RectLandmark>(landmarkName)); // invisible landmark
		} else {
			cv::Rect_<float> rect(
					relativePosition.x * imageWidth,
					relativePosition.y * imageHeight,
					relativePosition.width * imageWidth,
					relativePosition.height * imageHeight);
			collection.insert(make_shared<RectLandmark>(landmarkName, rect));
		}
	} // Note: else { throw... } ? This should never happen in regular operation, now that we removed get() ?
	return collection;
}

path BobotLandmarkSource::getName() const {
	if (index < 0)
		throw runtime_error("BobotLandmarkSource: invalid index");
	const auto iterator = index2name.find(static_cast<size_t>(index));
	if (iterator == index2name.end())
		throw runtime_error("BobotLandmarkSource: invalid index");
	return iterator->second;
}

} /* namespace imageio */
