/*
 * IbugLandmarkFormatParser.cpp
 *
 *  Created on: 05.11.2013
 *      Author: Patrik Huber
 */

#include "imageio/IbugLandmarkFormatParser.hpp"
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

IbugLandmarkFormatParser::~IbugLandmarkFormatParser() {}

const map<path, LandmarkCollection> IbugLandmarkFormatParser::read(path landmarkFilePath)
{
	map<path, LandmarkCollection> allLandmarks;
	LandmarkCollection landmarks;

	ifstream ifLM(landmarkFilePath.string());
	string line;
	getline(ifLM, line); // Skip the first 3 lines, they're header lines
	getline(ifLM, line);
	getline(ifLM, line);

	int landmarkId = 1; // The landmarks are ordered in the .pts file
	while(getline(ifLM, line))
	{
		if (line == "}") {
			break;
		}
		stringstream ssLine(line);
		Vec3f position(0.0f, 0.0f, 0.0f);
		if (!(ssLine >> position[0] >> position[1])) {
			throw std::runtime_error("Landmark format error while parsing a line.");
		}
		
		string tlmsName = iBugToTlmsName(landmarkId);
		bool visible = true; // In comparison to the original LFPW, this information is not available anymore.
		shared_ptr<Landmark> lm = make_shared<ModelLandmark>(tlmsName, position, visible);
		if (lm->getName().length() != 0) { // Todo: Find better solution
			landmarks.insert(lm);
		}
		
		
		++landmarkId;
	}
	path imageName = landmarkFilePath.stem();
	allLandmarks.insert(make_pair(imageName, landmarks));
	return allLandmarks;
}

map<int, string> IbugLandmarkFormatParser::iBugLmMapping;

std::string IbugLandmarkFormatParser::iBugToTlmsName(int iBugId)
{ // Todo: Remove that, use the file
	if (iBugLmMapping.empty()) {
		// TODO: Check all the landmarks with the .tlms file open in notepad++
		iBugLmMapping.insert(make_pair(18, "right.eyebrow.bend.lower"));
		iBugLmMapping.insert(make_pair(22, "right.eyebrow.inner_lower"));
		iBugLmMapping.insert(make_pair(27, "left.eyebrow.bend.lower"));
		iBugLmMapping.insert(make_pair(23, "left.eyebrow.inner_lower"));
		// eyebrow centers
		iBugLmMapping.insert(make_pair(37, "right.eye.corner_outer"));
		iBugLmMapping.insert(make_pair(46, "left.eye.corner_outer"));
		iBugLmMapping.insert(make_pair(40, "right.eye.corner_inner"));
		iBugLmMapping.insert(make_pair(43, "left.eye.corner_inner"));
		// left/right nose
		iBugLmMapping.insert(make_pair(31, "center.nose.tip"));
		iBugLmMapping.insert(make_pair(49, "right.lips.corner"));
		iBugLmMapping.insert(make_pair(55, "left.lips.corner"));
		iBugLmMapping.insert(make_pair(52, "center.lips.upper.outer"));
		iBugLmMapping.insert(make_pair(63, "center.lips.upper.inner"));
		iBugLmMapping.insert(make_pair(67, "center.lips.lower.inner"));
		iBugLmMapping.insert(make_pair(58, "center.lips.lower.outer"));
		iBugLmMapping.insert(make_pair( 9, "center.chin.tip"));
	}
	const auto tlmsName = iBugLmMapping.find(iBugId); // maybe use .at() and let it throw, or just throw/log by ourselves
	if(tlmsName != iBugLmMapping.end())
		return tlmsName->second;
	else
		return string("");	// Todo: That's not so nice, do this differently.
}


} /* namespace imageio */
