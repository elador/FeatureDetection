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

using boost::algorithm::trim;
using boost::algorithm::starts_with;
using std::ifstream;
using std::stringstream;
using std::getline;

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
	 * Parses a line from an LFW .lst file and returns the facebox.
	 * TODO: I think .lst files are actually from MR. They could also contain
	 *       more landmarks, but at the moment we don't have that use case.
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
	 * Parse a line of a tlms file and return a Landmark.
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
			throw std::runtime_error("Landmark parsing format error, use tlms");
		}
		if ( !(sstrLine >> fPos[2]) )
			fPos[2] = 0;

		return Landmark(name, fPos, bVisible > 0);
	}

};

} /* namespace imageio */
#endif /* LANDMARKSHELPER_HPP_ */
