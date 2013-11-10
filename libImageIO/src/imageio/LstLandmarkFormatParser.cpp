/*
 * LstLandmarkFormatParser.cpp
 *
 *  Created on: 09.11.2013
 *      Author: Patrik Huber
 */

#include "imageio/LstLandmarkFormatParser.hpp"
#include "imageio/RectLandmark.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include <stdexcept>
#include <utility>
#include <fstream>

using std::copy;
using std::sort;
using std::runtime_error;
using std::make_pair;
using boost::algorithm::trim;
using boost::algorithm::starts_with;
using boost::filesystem::path;
using std::ifstream;
using std::stringstream;
using std::make_shared;
using std::pair;

namespace imageio {

const map<path, LandmarkCollection> LstLandmarkFormatParser::read(path landmarkFilePath)
{
	map<path, LandmarkCollection> allLandmarks;

	ifstream ifLM(landmarkFilePath.string());
	string line;
	
	while(getline(ifLM, line))
	{
		LandmarkCollection landmarks;
		stringstream ssLine(line);
		path imageName;
		int x_tl, y_tl, x_br, y_br;
		if (!(ssLine >> imageName >> x_tl >> y_tl >> x_br >> y_br)) {
			throw std::runtime_error("Landmark format error while parsing a line.");
		}
		shared_ptr<Landmark> lm = make_shared<RectLandmark>("face", Rect(x_tl, y_tl, x_br-x_tl, y_br-y_tl)); // visible by default
		landmarks.insert(lm);
		allLandmarks.insert(make_pair(imageName.stem(), landmarks));
	}

	return allLandmarks;
}

} /* namespace imageio */
