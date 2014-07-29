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
		LandmarkCollection eyes;
		
		float le_x = lexical_cast<float>(tokens[1]);
		float le_y = lexical_cast<float>(tokens[2]);
		float re_x = lexical_cast<float>(tokens[3]);
		float re_y = lexical_cast<float>(tokens[4]);
		shared_ptr<Landmark> le = make_shared<ModelLandmark>("le", le_x, le_y);
		eyes.insert(le);
		shared_ptr<Landmark> re = make_shared<ModelLandmark>("re", re_x, re_y);
		eyes.insert(re);
		
		allLandmarks.insert(make_pair(imageName, eyes));
	}

	return allLandmarks;
}

} /* namespace imageio */
