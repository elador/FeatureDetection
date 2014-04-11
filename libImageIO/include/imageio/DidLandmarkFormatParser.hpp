/*
 * DidLandmarkFormatParser.hpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#ifndef DIDLANDMARKFORMATPARSER_HPP_
#define DIDLANDMARKFORMATPARSER_HPP_

#include "imageio/LandmarkFormatParser.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageio/ModelLandmark.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>

namespace imageio {

/**
 * Takes the path to a .did file as input and returns a map with one entry containing
 * a LandmarkCollection (TODO in tlms format) with all the landmarks found.
 */
class DidLandmarkFormatParser : public LandmarkFormatParser {
public:

	virtual ~DidLandmarkFormatParser();

	/**
	 * Reads the landmark data for one single image and returns all its landmarks (TODO in tlms format).
	 *
	 * @param[in] landmarkFilePath A path to a .did file.
	 * @return A map with one entry containing the basename of the file and
	 *         all the landmarks that are present (TODO in tlms format).
	 */
	const std::map<boost::filesystem::path, LandmarkCollection> read(boost::filesystem::path landmarkFilePath);

private:
	static std::map<int, std::string> didLmMapping;	///< Contains a mapping from the .did Surrey 3DMM to tlms landmark names
	
	/**
	 * Opens and parses a .did file and returns a collection of all the landmarks it contains.
	 *
	 * @param[in] filename The file name of the .did file to parse.
	 * @return A collection of all the landmarks.
	 */
	LandmarkCollection readFromDidFile(const std::string& filename);

	/**
	 * Parse a line of a .did file and return a Landmark.
	 *
	 * @param[in] line The line with the landmark information to parse.
	 * @return A Landmark object.
	 */
	std::shared_ptr<ModelLandmark> readFromDidLine(const std::string& line);
	
public:
	static std::string didToTlmsName(int didVertexId);
};

} /* namespace imageio */
#endif /* DIDLANDMARKFORMATPARSER_HPP_ */
