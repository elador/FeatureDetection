/*
 * SimpleRectLandmarkFormatParser.hpp
 *
 *  Created on: 18.07.2014
 *      Author: Patrik Huber
 */

#include "imageio/SimpleRectLandmarkFormatParser.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include <stdexcept>
#include <utility>
#include <fstream>

using cv::Vec3f;
using boost::filesystem::path;
using boost::algorithm::trim;
using std::map;
using std::string;
using std::getline;
using std::shared_ptr;
using std::ifstream;
using std::stringstream;
using std::make_pair;
using std::make_shared;


namespace imageio {

const map<path, LandmarkCollection> SimpleRectLandmarkFormatParser::read(path landmarkFilePath)
{
	map<path, LandmarkCollection> lmcoll;
	lmcoll.insert(make_pair(landmarkFilePath.stem(), parseFile(landmarkFilePath.string())));
	return lmcoll;
}

LandmarkCollection SimpleRectLandmarkFormatParser::parseFile(const string& filename) {
	ifstream landmarksFile(filename);
	string line;
	LandmarkCollection landmarks;

	while(getline(landmarksFile, line))
	{
		trim(line); // removes trailing and leading whitespaces (and other stuff?)
		if (!line.empty())
		{
			shared_ptr<RectLandmark> lm = parseLine(line);
			landmarks.insert(lm);
		}
	}
	return landmarks;
}

shared_ptr<RectLandmark> SimpleRectLandmarkFormatParser::parseLine(const string& line) {
	stringstream lineStream(line);
	string name;
	float topLeftX, topLeftY, w, h;

	if (!(lineStream >> name >> topLeftX >> topLeftY >> w >> h)) {
		throw std::runtime_error("Landmark parsing format error. Line should be 'landmarkName topLeftX topLeftY width height'.");
	}
	cv::Rect box(topLeftX, topLeftY, w, h);
	return make_shared<RectLandmark>(name, box); // visible will be true
}

} /* namespace imageio */
