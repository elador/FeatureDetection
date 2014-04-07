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
#include "imageio/ModelLandmark.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>

namespace imageio {

/**
 * Takes the path to a .tlms file as input and returns a map with one
 * entry containing a LandmarkCollection with all the landmarks found.
 */
class TlmsLandmarkFormatParser : public LandmarkFormatParser {
public:

	virtual ~TlmsLandmarkFormatParser();

	/**
	 * Reads the landmark data for one single image and returns all its landmarks.
	 *
	 * @param[in] landmarkFilePath A path to a .tlms file.
	 * @return A map with one entry containing the basename of the
	 *         file and all the landmarks that are present.
	 */
	const std::map<boost::filesystem::path, LandmarkCollection> read(boost::filesystem::path landmarkFilePath);

private:
	/**
	 * Opens and parses a .tlms file and returns a collection of all the landmarks it contains.
	 *
	 * @param[in] filename The file name of the .tlms file to parse.
	 * @return A collection of all the landmarks.
	 */
	LandmarkCollection readFromTlmsFile(const std::string& filename);

	/**
	 * Parse a line of a .tlms file and return a Landmark.
	 *
	 * @param[in] line The line with the landmark information to parse.
	 * @return A Landmark object.
	 */
	std::shared_ptr<ModelLandmark> readFromTlmsLine(const std::string& line);
	
};

} /* namespace imageio */
#endif /* TLMSLANDMARKFORMATPARSER_HPP_ */
