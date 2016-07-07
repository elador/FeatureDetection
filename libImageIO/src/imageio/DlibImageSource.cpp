/*
 * DlibImageSource.cpp
 *
 *  Created on: 31.07.2015
 *      Author: poschmann
 */

#include "imageio/DlibImageSource.hpp"
#include "imageio/RectLandmark.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/property_tree/xml_parser.hpp"
#include <stdexcept>

using boost::property_tree::ptree;
using std::pair;
using std::string;

namespace imageio {

DlibImageSource::DlibImageSource(const string& filename) : LabeledImageSource(filename) {
	directory = boost::filesystem::path(filename).parent_path();
	boost::property_tree::xml_parser::read_xml(filename, info);
	const ptree& images = info.get_child("dataset.images");
	auto range = images.equal_range("image");
	imagesBegin = range.first;
	imagesEnd = range.second;
	imagesNext = imagesBegin;
}

void DlibImageSource::reset() {
	imagesNext = imagesBegin;
}

bool DlibImageSource::next() {
	if (imagesNext == imagesEnd)
		return false;
	string imageFilename = imagesNext->second.get<string>("<xmlattr>.file");
	boost::filesystem::path imageFilepath = directory;
	imageFilepath /= imageFilename;
	name = imageFilepath;
	image = cv::imread(imageFilepath.string(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
		throw std::runtime_error("image '" + imageFilepath.string() + "' could not be loaded");
	landmarks.clear();
	int objectCount = 0;
	int ignoreCount = 0;
	auto boxesRange = imagesNext->second.equal_range("box");
	for (auto it = boxesRange.first; it != boxesRange.second; ++it) {
		int top = it->second.get<int>("<xmlattr>.top");
		int left = it->second.get<int>("<xmlattr>.left");
		int width = it->second.get<int>("<xmlattr>.width");
		int height = it->second.get<int>("<xmlattr>.height");
		boost::optional<bool> ignore = it->second.get_optional<bool>("<xmlattr>.ignore");
		string name;
		if (ignore && *ignore) // box should be ignored (is neither positive nor negative sample)
			name = "ignore" + std::to_string(ignoreCount++);
		else // box should not be ignored (is considered positive sample)
			name = "object" + std::to_string(objectCount++);
		landmarks.insert(std::make_shared<RectLandmark>(name, cv::Rect(left, top, width, height)));
	}
	++imagesNext;
	return true;
}

const cv::Mat DlibImageSource::getImage() const {
	return image;
}

boost::filesystem::path DlibImageSource::getName() const {
	return name;
}

const LandmarkCollection DlibImageSource::getLandmarks() const {
	return landmarks;
}

std::vector<boost::filesystem::path> DlibImageSource::getNames() const {
	std::vector<boost::filesystem::path> names;
	for (auto it = imagesBegin; it != imagesEnd; ++it) {
		string imageFilename = it->second.get<string>("<xmlattr>.file");
		boost::filesystem::path imageFilepath = directory;
		imageFilepath /= imageFilename;
		names.push_back(imageFilepath);
	}
	return names;
}

} /* namespace imageio */
