/*
 * TlmsLandmarkFormatParser.cpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/TlmsLandmarkFormatParser.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include <stdexcept>
#include <utility>
#include <fstream>

using cv::Vec3f;
using boost::filesystem::path;
using boost::algorithm::starts_with;
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

TlmsLandmarkFormatParser::~TlmsLandmarkFormatParser() {}

const map<path, LandmarkCollection> TlmsLandmarkFormatParser::read(path landmarkFilePath)
{
	map<path, LandmarkCollection> lmcoll;
	lmcoll.insert(make_pair(landmarkFilePath, readFromTlmsFile(landmarkFilePath.string())));
	return lmcoll;
}

LandmarkCollection TlmsLandmarkFormatParser::readFromTlmsFile(const string& filename) {
	ifstream ifLM(filename);
	string strLine;
	LandmarkCollection listLM;

	while(getline(ifLM, strLine))
	{
		trim(strLine);
		// allow comments
		if (!strLine.empty() && !starts_with(strLine, "#") && !starts_with(strLine, "//"))
		{
			shared_ptr<ModelLandmark> lm = readFromTlmsLine(strLine);
			listLM.insert(lm);
		}
	}
	return listLM;
}

shared_ptr<ModelLandmark> TlmsLandmarkFormatParser::readFromTlmsLine(const string& line) {
	stringstream sstrLine(line);
	string name;
	Vec3f fPos(0.0f, 0.0f, 0.0f);
	int bVisible = 1;

	if ( !(sstrLine >> name >> bVisible >> fPos[0] >> fPos[1]) ) {
		throw std::runtime_error("Landmark parsing format error, use .tlms");
	}
	if ( !(sstrLine >> fPos[2]) )
		fPos[2] = 0;

	return make_shared<ModelLandmark>(name, fPos, bVisible > 0);
}

} /* namespace imageio */
