/*
 * LandmarksHelper.hpp
 *
 *  Created on: 23.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LANDMARKSHELPER_HPP_
#define LANDMARKSHELPER_HPP_

#include "imageio/Landmark.hpp"
#include "imageio/FaceBoxLandmark.hpp"
#include "imageio/LandmarksCollection.hpp"
#include "boost/algorithm/string.hpp"
#include <fstream>
#include <map>
#include <string>

using boost::algorithm::trim;
using boost::algorithm::starts_with;
using std::ifstream;
using std::stringstream;
using std::getline;
using std::map;
using std::string;
using std::make_pair;

namespace imageio {

/**
 * A helper class to read from all kinds of different landmark files.
 *
 * Note: Later on, this class could also provide functions to write landmark files.
 */
class LandmarksHelper {
public:
	/**
	 * Opens and parses a file containing landmarks and returns them. Choses the appropriate parsing
	 * function according to the file-ending.
	 * TODO
	 *
	 * @param[in] filename The file name of the file to parse.
	 * @return A collection of all the landmarks.
	 */
	//static LandmarksCollection readFromFile(const string& filename);

	/**
	 * Opens and parses a .tlms file and returns a collection of all the landmarks it contains.
	 *
	 * @param[in] filename The file name of the .tlms file to parse.
	 * @return A collection of all the landmarks.
	 */
	static LandmarksCollection readFromTlmsFile(const string& filename) {
		ifstream ifLM(filename);
		string strLine;
		LandmarksCollection listLM;

		while(getline(ifLM, strLine))
		{
			boost::algorithm::trim(strLine);
			// allow comments
			if ( !strLine.empty() && !starts_with(strLine, "#") && !starts_with(strLine, "//") )
			{
				Landmark lm = LandmarksHelper::readFromTlmsLine(strLine);
				listLM.insert(lm);
			}
		}
		return listLM;
	}

	/**
	 * Opens and parses a .did file and returns a collection of all the landmarks it contains.
	 *
	 * @param[in] filename The file name of the .did file to parse.
	 * @return A collection of all the landmarks.
	 */
	static LandmarksCollection readFromDidFile(const string& filename) {
		ifstream ifLM(filename);
		string strLine;
		LandmarksCollection listLM;

		while(getline(ifLM, strLine))
		{
			boost::algorithm::trim(strLine);
			// allow comments
			if ( !strLine.empty() && !starts_with(strLine, "#") && !starts_with(strLine, "//") )
			{
				Landmark lm = LandmarksHelper::readFromDidLine(strLine);
				listLM.insert(lm);
			}
		}
		return listLM;
	}

	/**
	 * Parses a line from an LFW .lst file and returns the facebox.
	 * TODO: I think .lst files are actually from MR. They could also contain
	 *       more landmarks, but at the moment we don't have that use-case.
	 *
	 * @param[in] line The line with the landmark information to parse.
	 * @return A FaceBoxLandmark object.
	 */
	static FaceBoxLandmark readFromLstLine(const string& line) {
		int l=0, r=0, b=0, t=0;	// TODO use floats?
		string buf;
		stringstream ss(line);
		ss >> buf;
		ss >> l;
		ss >> t;
		ss >> r;
		ss >> b;
		return FaceBoxLandmark("facebox", Vec3f(l+(r-l)/2.0f, t+(b-t)/2.0f, 0.0f), r-l, b-t);
	}

private:
	/**
	 * Parse a line of a .tlms file and return a Landmark.
	 *
	 * @param[in] line The line with the landmark information to parse.
	 * @return A Landmark object.
	 */
	static Landmark readFromTlmsLine(const string& line) {
		stringstream sstrLine(line);
		string name;
		Vec3f fPos(0.0f, 0.0f, 0.0f);
		int bVisible = 1;

		if ( !(sstrLine >> name >> bVisible >> fPos[0] >> fPos[1]) ) {
			throw std::runtime_error("Landmark parsing format error, use .tlms");
		}
		if ( !(sstrLine >> fPos[2]) )
			fPos[2] = 0;

		return Landmark(name, fPos, bVisible > 0);
	}

	/**
	 * Parse a line of a .did file and return a Landmark.
	 *
	 * @param[in] line The line with the landmark information to parse.
	 * @return A Landmark object.
	 */
	static Landmark readFromDidLine(const string& line) {
		stringstream sstrLine(line);
		string name;
		Vec3f fPos(0.0f, 0.0f, 0.0f);
		float fVertexNumber;

		if ( !(sstrLine >> fPos[0] >> fPos[1] >> fVertexNumber) ) {
			throw std::runtime_error("Landmark parsing format error, use .did");
		}
		fPos[2] = 0;
		int vertexNumber = boost::lexical_cast<int>(fVertexNumber);
		return Landmark(name, fPos, true);
	}

	static map<int, string> didLmMapping;	///< Contains a mapping from the .did Surrey 3DMM to tlms landmark names

	static string didToTlmsName(int didVertexId) {
		if (didLmMapping.empty()) {
			didLmMapping.insert(make_pair( 177, "right.eye.corner_outer"));
			didLmMapping.insert(make_pair( 610, "left.eye.corner_outer"));
			didLmMapping.insert(make_pair( 114, "center.nose.tip"));
			//didLmMapping.insert(make_pair(  35, "center.chin.tip"));	// I think the .did Chin-tip is about 1cm farther up
			didLmMapping.insert(make_pair( 181, "right.eye.corner_inner")); // TODO double-check
			didLmMapping.insert(make_pair(1125, ""));
			didLmMapping.insert(make_pair(1180, ""));
			didLmMapping.insert(make_pair( 614, ""));
			didLmMapping.insert(make_pair(2368, ""));
			didLmMapping.insert(make_pair(2425, ""));
			didLmMapping.insert(make_pair( 438, ""));
			didLmMapping.insert(make_pair( 398, ""));
			didLmMapping.insert(make_pair( 812, ""));
			didLmMapping.insert(make_pair( 329, ""));
			didLmMapping.insert(make_pair( 423, ""));
			didLmMapping.insert(make_pair( 442, ""));
			didLmMapping.insert(make_pair( 411, ""));
			didLmMapping.insert(make_pair( 225, ""));
			didLmMapping.insert(make_pair( 157, ""));
			didLmMapping.insert(make_pair( 233, ""));
			didLmMapping.insert(make_pair(  79, ""));
			didLmMapping.insert(make_pair( 658, ""));
			didLmMapping.insert(make_pair( 590, ""));
			didLmMapping.insert(make_pair( 666, ""));
			didLmMapping.insert(make_pair( 516, ""));
			didLmMapping.insert(make_pair( 332, ""));
			didLmMapping.insert(make_pair( 295, ""));
			didLmMapping.insert(make_pair( 379, ""));
			didLmMapping.insert(make_pair( 270, ""));
			didLmMapping.insert(make_pair( 440, ""));
			didLmMapping.insert(make_pair( 314, ""));
			didLmMapping.insert(make_pair( 416, ""));
			didLmMapping.insert(make_pair( 404, ""));
			didLmMapping.insert(make_pair( 263, ""));
			didLmMapping.insert(make_pair( 735, ""));
			didLmMapping.insert(make_pair( 828, ""));
			didLmMapping.insert(make_pair( 817, ""));
			didLmMapping.insert(make_pair( 692, ""));
			didLmMapping.insert(make_pair( 359, ""));
			didLmMapping.insert(make_pair( 776, ""));
		}
	}

};

} /* namespace imageio */
#endif /* LANDMARKSHELPER_HPP_ */
