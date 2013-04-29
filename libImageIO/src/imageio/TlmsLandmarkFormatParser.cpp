/*
 * TlmsLandmarkFormatParser.cpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/TlmsLandmarkFormatParser.hpp"
#include "imageio/Landmark.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include <stdexcept>
#include <utility>
#include <fstream>

using std::copy;
using std::sort;
using std::runtime_error;
using std::make_pair;
using boost::algorithm::trim;
using boost::algorithm::starts_with;
using std::ifstream;
using std::stringstream;

namespace imageio {

TlmsLandmarkFormatParser::~TlmsLandmarkFormatParser() {}

const LandmarkCollection TlmsLandmarkFormatParser::read(vector<string> landmarkData)
{
	LandmarkCollection lmcoll;
	for (auto line : landmarkData) {
		// TODO: Add checking of #, //, ... (see readFromTlmsFile(...) below).
		lmcoll.insert(readFromTlmsLine(line));
	}
	return lmcoll;
}

LandmarkCollection TlmsLandmarkFormatParser::readFromTlmsFile(const string& filename) {
	ifstream ifLM(filename);
	string strLine;
	LandmarkCollection listLM;

	while(getline(ifLM, strLine))
	{
		boost::algorithm::trim(strLine);
		// allow comments
		if ( !strLine.empty() && !starts_with(strLine, "#") && !starts_with(strLine, "//") )
		{
			Landmark lm = readFromTlmsLine(strLine);
			listLM.insert(lm);
		}
	}
	return listLM;
}

Landmark TlmsLandmarkFormatParser::readFromTlmsLine(const string& line) {
	stringstream sstrLine(line);
	string name;
	Vec3f fPos(0.0f, 0.0f, 0.0f);
	int bVisible = 1;

	if ( !(sstrLine >> name >> bVisible >> fPos[0] >> fPos[1]) ) {
		throw std::runtime_error("Landmark parsing format error, use .tlms");
	}
	if ( !(sstrLine >> fPos[2]) )
		fPos[2] = 0;

	return Landmark(name, fPos, cv::Size2f(), bVisible > 0);
}

} /* namespace imageio */
