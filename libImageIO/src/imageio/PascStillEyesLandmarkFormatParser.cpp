/*
 * PascStillEyesLandmarkFormatParser.cpp
 *
 *  Created on: 25.07.2014
 *      Author: Patrik Huber
 */

#include "imageio/PascStillEyesLandmarkFormatParser.hpp"
#include "imageio/ModelLandmark.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include <utility>
#include <fstream>

using cv::Vec3f;
using boost::filesystem::path;
using boost::lexical_cast;
using std::make_pair;
using std::ifstream;
using std::make_shared;
using std::map;
using std::getline;
using std::vector;
using std::string;
using std::shared_ptr;

namespace imageio {

const map<path, LandmarkCollection> PascStillEyesLandmarkFormatParser::read(path landmarkFilePath)
{
	map<path, LandmarkCollection> allLandmarks;

	ifstream csvFile(landmarkFilePath.string());
	string line;
	getline(csvFile, line); // The header line, we skip it
	
	while(getline(csvFile, line))
	{
		vector<string> tokens;
		boost::trim_right_if(line, boost::is_any_of("\r")); // Windows line-endings are \r\n, Linux only \n. Thus, when a file has been created on windows and is read on linux, we need to remove the trailing \r.
		boost::split(tokens, line, boost::is_any_of(","));
		path imageName = tokens[0];
		LandmarkCollection landmarks;
		for (int landmarkId = 0; landmarkId < 76; ++landmarkId) {
			float x = lexical_cast<float>(tokens[landmarkId * 2 + 2]);
			float y = lexical_cast<float>(tokens[landmarkId * 2 + 2 + 1]);
			bool available = true; // "Unavailable points" are points that are obscured by other facial features. (i.e. self-occlusion. Occlusions by hair or glasses are marked as visible)
			if (tokens[landmarkId * 2 + 2] == "0" && tokens[landmarkId * 2 + 2 + 1] == "0") {
				available = false;
			}
			shared_ptr<Landmark> lm = make_shared<ModelLandmark>(lexical_cast<string>(landmarkId), Vec3f(x, y, 0.0f), available);
			landmarks.insert(lm);
		}
		
		allLandmarks.insert(make_pair(imageName, landmarks));
	}

	return allLandmarks;
}

} /* namespace imageio */
