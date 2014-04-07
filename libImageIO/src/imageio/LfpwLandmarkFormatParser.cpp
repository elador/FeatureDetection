/*
 * LfpwLandmarkFormatParser.cpp
 *
 *  Created on: 04.11.2013
 *      Author: Patrik Huber
 */

#include "imageio/LfpwLandmarkFormatParser.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include <stdexcept>
#include <utility>
#include <fstream>

using std::map;
using std::string;
using std::getline;
using std::vector;
//using boost::algorithm::trim;
//using boost::algorithm::starts_with;
using boost::filesystem::path;
using std::ifstream;
using std::shared_ptr;
using std::make_shared;
using std::pair;
using std::make_pair;


namespace imageio {

LfpwLandmarkFormatParser::~LfpwLandmarkFormatParser() {}

const map<path, LandmarkCollection> LfpwLandmarkFormatParser::read(path landmarkFilePath)
{
	map<path, LandmarkCollection> allLandmarks;

	ifstream ifLM(landmarkFilePath.string());
	string line;
	string header;
	getline(ifLM, header); // Skip the first line, it's the header

	while(getline(ifLM, line))
	{
		// we split here, because we want to skip the line if it's not the 'average' labeling
		vector<string> tokens;
		boost::split(tokens, line, boost::is_any_of("\t"));
		if (tokens[1] != "average") {
			continue;
		}
		pair<path, LandmarkCollection> landmarks = readLine(tokens, header);
		allLandmarks.insert(landmarks);
	}

	return allLandmarks;
}

pair<path, LandmarkCollection> LfpwLandmarkFormatParser::readLine(const vector<string>& line, const string header)
{
	path imageName = path(line[0]).filename();
	LandmarkCollection landmarks;

	vector<string> headerTokens; // split the header
	boost::split(headerTokens, header, boost::is_any_of("\t"));
	unsigned int offset = -1; // The offset to go from the landmark number to the corresponding entry in the vector or line
	for (unsigned int i = 1; i <= 35; ++i) { // i corresponds to the landmark number of LFPW (see their picture)
		string landmarkName = headerTokens[3*i+offset].substr(0, headerTokens[3*i+offset].length()-2); // cuts off the last two characters ('_x')
		string tlmsName = lfpwToTlmsName(landmarkName);
		bool visible = true; // Todo: Could add "obscured" to Landmark class
		if (boost::lexical_cast<int>(line[3*i+offset+2]) == 2 || boost::lexical_cast<int>(line[3*i+offset+2]) == 3) {
			visible = false;
		}
		cv::Vec3f position(boost::lexical_cast<float>(line[3*i+offset]), boost::lexical_cast<float>(line[3*i+offset+1]), 0.0f);
		shared_ptr<Landmark> lm = make_shared<ModelLandmark>(tlmsName, position, visible);
		if (lm->getName().length() != 0) { // Todo: Find better solution
			landmarks.insert(lm);
		}
	}
	
	return make_pair(imageName, landmarks);
}

map<string, string> LfpwLandmarkFormatParser::lfpwLmMapping;

string LfpwLandmarkFormatParser::lfpwToTlmsName(string lfpwName)
{ // Todo: Remove that, use the file
	if (lfpwLmMapping.empty()) {
		// TODO: Check all the landmarks with the .tlms file open in notepad++
		lfpwLmMapping.insert(make_pair("left_eyebrow_out", "right.eyebrow.bend.lower")); // 1 Todo check tlms
		lfpwLmMapping.insert(make_pair("left_eyebrow_in", "right.eyebrow.inner_lower")); // 3 Todo check tlms
		lfpwLmMapping.insert(make_pair("right_eyebrow_out", "left.eyebrow.bend.lower")); // 2 Todo check tlms
		lfpwLmMapping.insert(make_pair("right_eyebrow_in", "left.eyebrow.inner_lower")); // 4 Todo check tlms
		//lfpwLmMapping.insert(make_pair("left_eyebrow_center_top", "")); // 5 Todo check tlms
		//lfpwLmMapping.insert(make_pair("left_eyebrow_center_bottom", "")); // 6 Todo check tlms
		//lfpwLmMapping.insert(make_pair("right_eyebrow_center_top", "")); // 7 Todo check tlms
		//lfpwLmMapping.insert(make_pair("right_eyebrow_center_bottom", "")); // 8 Todo check tlms
		lfpwLmMapping.insert(make_pair("left_eye_out", "right.eye.corner_outer")); // 9
		lfpwLmMapping.insert(make_pair("right_eye_out", "left.eye.corner_outer")); // 10
		lfpwLmMapping.insert(make_pair("left_eye_in", "right.eye.corner_inner")); // 11
		lfpwLmMapping.insert(make_pair("left_eye_center_top", "right.eye.top")); // 13 check
		lfpwLmMapping.insert(make_pair("left_eye_center_bottom", "right.eye.bottom")); // 14 check
		lfpwLmMapping.insert(make_pair("right_eye_in", "left.eye.corner_inner")); // 12
		lfpwLmMapping.insert(make_pair("right_eye_center_top", "left.eye.top")); // 15 check
		lfpwLmMapping.insert(make_pair("right_eye_center_bottom", "left.eye.bottom")); // 16 check
		// 17 left_eye_pupil
		// 18 right_eye_pupil
		// 19 left_nose_out
		// 20 right_nose_out
		lfpwLmMapping.insert(make_pair("nose_center_top", "center.nose.tip")); // 21
		// 22 nose_center_bottom
		lfpwLmMapping.insert(make_pair("left_mouth_out", "right.lips.corner")); // 23
		lfpwLmMapping.insert(make_pair("right_mouth_out", "left.lips.corner")); // 24
		lfpwLmMapping.insert(make_pair("mouth_center_top_lip_top", "center.lips.upper.outer")); // 25
		lfpwLmMapping.insert(make_pair("mouth_center_top_lip_bottom", "center.lips.upper.inner")); // 26
		lfpwLmMapping.insert(make_pair("mouth_center_bottom_lip_top", "center.lips.lower.inner")); // 27
		lfpwLmMapping.insert(make_pair("mouth_center_bottom_lip_bottom", "center.lips.lower.outer")); // 28
		// 29 left_ear_top - the ear landmarks are not shown on the LFPW images
		// 30 right_ear_top
		// 31 left_ear_bottom
		// 32 right_ear_bottom
		// 33 left_ear_canal
		// 34 right_ear_canal
		lfpwLmMapping.insert(make_pair("chin", "center.chin.tip")); // 35 (really? in the LFPW image it's 29...) TODO check that, visualize the Landmark on images!
	}
	const auto tlmsName = lfpwLmMapping.find(lfpwName); // maybe use .at() and let it throw, or just throw/log by ourselves
	if(tlmsName != lfpwLmMapping.end())
		return tlmsName->second;
	else
		return string("");	// Todo: That's not so nice, do this differently.
}


} /* namespace imageio */
