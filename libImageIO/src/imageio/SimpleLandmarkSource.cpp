/*
 * SimpleLandmarkSource.cpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#include "imageio/SimpleLandmarkSource.hpp"
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

namespace imageio {

const string SimpleLandmarkSource::landmarkName = "target";

SimpleLandmarkSource::SimpleLandmarkSource(const string& filename) : positions(), index(-1) {
	string line;
	char separator;
	std::ifstream file(filename.c_str());
	if (!file.is_open())
		throw invalid_argument("SimpleLandmarkSource: file \"" + filename + "\" cannot be opened");
	Rect_<float> position;
	while (file.good()) {
		if (!std::getline(file, line))
			break;
		// read values from line
		std::istringstream lineStream(line);
		bool nonSpaceSeparator = !std::all_of(line.begin(), line.end(), [](char ch) {
			return std::isdigit(ch) || std::isspace(ch) || '-' == ch || '.' == ch;
		});
		lineStream >> position.x;
		if (nonSpaceSeparator)
			lineStream >> separator;
		lineStream >> position.y;
		if (nonSpaceSeparator)
			lineStream >> separator;
		lineStream >> position.width;
		if (nonSpaceSeparator)
			lineStream >> separator;
		lineStream >> position.height;

		positions.push_back(position);
	}
}

void SimpleLandmarkSource::reset() {
	index = -1;
}

bool SimpleLandmarkSource::next() {
	index++;
	return index < static_cast<int>(positions.size());
}

LandmarkCollection SimpleLandmarkSource::get(const path& imagePath) {
	index = std::stoi(imagePath.string());
	return getLandmarks();
}

LandmarkCollection SimpleLandmarkSource::getLandmarks() const {
	LandmarkCollection collection;
	if (index >= 0 && index < static_cast<int>(positions.size())) {
		const Rect_<float>& position = positions[index];
		if (position.width > 0 && position.height > 0)
			collection.insert(make_shared<RectLandmark>(landmarkName, position));
		else // landmark is invisible
			collection.insert(make_shared<RectLandmark>(landmarkName));
	}
	return collection;
}

path SimpleLandmarkSource::getName() const {
	return path(std::to_string(index));
}

} /* namespace imageio */
