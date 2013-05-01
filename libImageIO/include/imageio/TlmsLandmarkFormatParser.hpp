/*
 * TlmsLandmarkFormatParser.hpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#ifndef TLMSLANDMARKFORMATPARSER_HPP_
#define TLMSLANDMARKFORMATPARSER_HPP_

#include "imageio/LandmarkFormatParser.hpp"
#include "imageio/LandmarkCollection.hpp"
#include <vector>
#include <string>
#include <map>

using std::vector;
using std::string;
using std::map;

namespace imageio {

/**
 * Takes one or several lines from a landmarks file for one image as 
 * input and returns a LandmarkCollection (TODO in tlms format) with all the landmarks found.
 */
class TlmsLandmarkFormatParser : public LandmarkFormatParser {
public:

	virtual ~TlmsLandmarkFormatParser();

	/**
	 * Reads the landmark data for one single image and returns all its landmarks (TODO in tlms format).
	 *
	 * @param[in] landmarkData One or several lines from a landmarks file for one image.
	 * @return All the landmarks that are present in the input (TODO in tlms format).
	 */
	const map<path, LandmarkCollection> read(path landmarkFilePath);

private:
	
	/**
	 * Opens and parses a .tlms file and returns a collection of all the landmarks it contains.
	 *
	 * @param[in] filename The file name of the .tlms file to parse.
	 * @return A collection of all the landmarks.
	 */
	LandmarkCollection readFromTlmsFile(const string& filename);

	/**
	 * Parse a line of a .tlms file and return a Landmark.
	 *
	 * @param[in] line The line with the landmark information to parse.
	 * @return A Landmark object.
	 */
	Landmark readFromTlmsLine(const string& line);
	
};

} /* namespace imageio */
#endif /* TLMSLANDMARKFORMATPARSER_HPP_ */
