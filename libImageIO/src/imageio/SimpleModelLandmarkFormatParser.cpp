/*
 * SimpleModelLandmarkFormatParser.hpp
 *
 *  Created on: 29.05.2014
 *      Author: Patrik Huber
 */

#include "imageio/SimpleModelLandmarkFormatParser.hpp"
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

const map<path, LandmarkCollection> SimpleModelLandmarkFormatParser::read(path landmarkFilePath)
{
	map<path, LandmarkCollection> lmcoll;
	lmcoll.insert(make_pair(landmarkFilePath.stem(), parseFile(landmarkFilePath.string())));
	return lmcoll;
}

LandmarkCollection SimpleModelLandmarkFormatParser::parseFile(const string& filename) {
	ifstream landmarksFile(filename);
	string line;
	LandmarkCollection landmarks;

	while(getline(landmarksFile, line))
	{
		trim(line); // removes trailing and leading whitespaces (and other stuff?)
		if (!line.empty())
		{
			shared_ptr<ModelLandmark> lm = parseLine(line);
			landmarks.insert(lm);
		}
	}
	return landmarks;
}

shared_ptr<ModelLandmark> SimpleModelLandmarkFormatParser::parseLine(const string& line) {
	stringstream lineStream(line);
	string name;
	float x, y;

	if (!(lineStream >> name >> x >> y)) {
		throw std::runtime_error("Landmark parsing format error. Line should be [name x y].");
	}

	return make_shared<ModelLandmark>(name, x, y); // visible will be true
}

} /* namespace imageio */
