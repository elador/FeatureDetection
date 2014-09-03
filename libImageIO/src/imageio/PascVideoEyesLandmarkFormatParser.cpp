/*
 * PascVideoEyesLandmarkFormatParser.cpp
 *
 *  Created on: 25.07.2014
 *      Author: Patrik Huber
 */

#include "imageio/PascVideoEyesLandmarkFormatParser.hpp"
#include "imageio/ModelLandmark.hpp"
#include "imageio/RectLandmark.hpp"
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

const map<path, LandmarkCollection> PascVideoEyesLandmarkFormatParser::read(path landmarkFilePath)
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
		
		float f_cx = lexical_cast<float>(tokens[1]);
		float f_cy = lexical_cast<float>(tokens[2]);
		float f_w = lexical_cast<float>(tokens[3]);
		float f_h = lexical_cast<float>(tokens[4]);
		shared_ptr<Landmark> face = make_shared<RectLandmark>("face", f_cx, f_cy, f_w, f_h);
		landmarks.insert(face);

		float le_x = lexical_cast<float>(tokens[6]);
		float le_y = lexical_cast<float>(tokens[7]);
		float re_x = lexical_cast<float>(tokens[8]);
		float re_y = lexical_cast<float>(tokens[9]);
		shared_ptr<Landmark> le = make_shared<ModelLandmark>("le", le_x, le_y);
		landmarks.insert(le);
		shared_ptr<Landmark> re = make_shared<ModelLandmark>("re", re_x, re_y);
		landmarks.insert(re);
		
		allLandmarks.insert(make_pair(imageName, landmarks));
	}

	return allLandmarks;
}

} /* namespace imageio */
