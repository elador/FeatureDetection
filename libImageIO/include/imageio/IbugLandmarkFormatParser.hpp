/*
 * IbugLandmarkFormatParser.hpp
 *
 *  Created on: 05.11.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef IBUGLANDMARKFORMATPARSER_HPP_
#define IBUGLANDMARKFORMATPARSER_HPP_

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
 * Takes landmarks in the form of .pts files from http://ibug.doc.ic.ac.uk/resources/300-W/ as
 * input and returns a map containing a LandmarkCollection in tlms format for every image.
 */
class IbugLandmarkFormatParser : public LandmarkFormatParser {
public:

	/**
	 * Reads the landmark data for one single image and returns all its landmarks.
	 *
	 * @param[in] landmarkFilePath A path to a .pts file.
	 * @return A map with one entry containing the basename of the file and
	 *         all 68 iBug landmarks.
	 */
	const std::map<boost::filesystem::path, LandmarkCollection> read(boost::filesystem::path landmarkFilePath);

};

} /* namespace imageio */
#endif /* IBUGLANDMARKFORMATPARSER_HPP_ */
