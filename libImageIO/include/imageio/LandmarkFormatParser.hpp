/*
 * LandmarkFormatParser.hpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#ifndef LANDMARKFORMATPARSER_HPP_
#define LANDMARKFORMATPARSER_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "imageio/LandmarkCollection.hpp"
#include "boost/filesystem.hpp"
#include <map>
#include <string>

using boost::filesystem::path;
using std::map;
using std::string;

namespace imageio {

/**
 * Takes a path to a landmarks file as input and returns a map with one LandmarkCollection
 * entry (TODO in tlms format) for each image that contains all the landmarks found.
 */
class LandmarkFormatParser {
public:

	virtual ~LandmarkFormatParser() {}

	/**
	 * Reads the landmark data from a file returns all its landmarks (TODO in tlms format).
	 *
	 * @param[in] landmarkFilePath A path to a file containing landmarks to one or several images/frames.
	 * @return All the landmarks that are present in the input (TODO in tlms format). The path is
	 *         stripped to only contain the basename.
	 */
	virtual const map<path, LandmarkCollection> read(path landmarkFilePath) = 0;
};

} /* namespace imageio */
#endif /* LANDMARKFORMATPARSER_HPP_ */
