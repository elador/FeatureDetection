/*
 * SdxLandmarkFormatParser.hpp
 *
 *  Created on: 05.08.2014
 *      Author: Patrik Huber
 */

#include "imageio/SdxLandmarkFormatParser.hpp"
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

const map<path, LandmarkCollection> SdxLandmarkFormatParser::read(path landmarkFilePath)
{
	map<path, LandmarkCollection> allLandmarks;

	ifstream csvFile(landmarkFilePath.string());
	string headerLine;
	getline(csvFile, headerLine); // The header line
	vector<string> headerTokens;
	boost::trim_right_if(headerLine, boost::is_any_of("\r")); // Windows line-endings are \r\n, Linux only \n. Thus, when a file has been created on windows and is read on linux, we need to remove the trailing \r.
	boost::split(headerTokens, headerLine, boost::is_any_of(" "));

	string line;
	while(getline(csvFile, line))
	{
		vector<string> tokens;
		boost::trim_right_if(line, boost::is_any_of("\r")); // Windows line-endings are \r\n, Linux only \n. Thus, when a file has been created on windows and is read on linux, we need to remove the trailing \r.
		boost::split(tokens, line, boost::is_any_of(" "));
		path imageName = tokens[0];
		LandmarkCollection landmarks;
		for (int landmarkId = 0; landmarkId < 5; ++landmarkId) {
			float x = lexical_cast<float>(tokens[landmarkId * 2 + 1]);
			float y = lexical_cast<float>(tokens[landmarkId * 2 + 1 + 1]);
			string landmarkName = headerTokens[landmarkId * 2 + 1];
			landmarkName = landmarkName.substr(0, landmarkName.length() - 1);
			shared_ptr<Landmark> lm = make_shared<ModelLandmark>(landmarkName, Vec3f(x, y, 0.0f), true);
			landmarks.insert(lm);
		}
		
		allLandmarks.insert(make_pair(imageName, landmarks));
	}

	return allLandmarks;
}

} /* namespace imageio */
