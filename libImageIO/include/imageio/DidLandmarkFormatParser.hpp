/*
 * DidLandmarkFormatParser.hpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#ifndef DIDLANDMARKFORMATPARSER_HPP_
#define DIDLANDMARKFORMATPARSER_HPP_

#include "imageio/LandmarkFormatParser.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
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
class DidLandmarkFormatParser : public LandmarkFormatParser {
public:

	virtual ~DidLandmarkFormatParser();

	/**
	 * Reads the landmark data for one single image and returns all its landmarks (TODO in tlms format).
	 *
	 * @param[in] landmarkData One or several lines from a landmarks file for one image.
	 * @return All the landmarks that are present in the input (TODO in tlms format).
	 */
	const map<path, LandmarkCollection> read(path landmarkFilePath);

private:
	map<int, string> didLmMapping;	///< Contains a mapping from the .did Surrey 3DMM to tlms landmark names
	
	/**
	 * Opens and parses a .did file and returns a collection of all the landmarks it contains.
	 *
	 * @param[in] filename The file name of the .did file to parse.
	 * @return A collection of all the landmarks.
	 */
	LandmarkCollection readFromDidFile(const string& filename);

	/**
	 * Parse a line of a .did file and return a Landmark.
	 *
	 * @param[in] line The line with the landmark information to parse.
	 * @return A Landmark object.
	 */
	Landmark readFromDidLine(const string& line);
	
	string didToTlmsName(int didVertexId);
};

} /* namespace imageio */
#endif /* DIDLANDMARKFORMATPARSER_HPP_ */
