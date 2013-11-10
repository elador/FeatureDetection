/*
 * LstLandmarkFormatParser.hpp
 *
 *  Created on: 09.11.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LSTLANDMARKFORMATPARSER_HPP_
#define LSTLANDMARKFORMATPARSER_HPP_

#include "imageio/LandmarkFormatParser.hpp"
#include "imageio/LandmarkCollection.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <map>

namespace imageio {

/**
 * Reads landmarks in the form of one .lst file that contains one line for each image.
 * A line consists of the path to the image, followed by 'x_topleft y_topleft x_bottomright y_bottomright'
 * of the face box (as integers).
 */
class LstLandmarkFormatParser : public LandmarkFormatParser {
public:

	/**
	 * Reads the landmark data for one single image and returns all its landmarks (TODO in tlms format).
	 *
	 * @param[in] landmarkFilePath A path to a .pts file.
	 * @return A map with one entry containing the basename of the file and
	 *         all the landmarks that are present (TODO in tlms format).
	 */
	const std::map<boost::filesystem::path, LandmarkCollection> read(boost::filesystem::path landmarkFilePath);

};

} /* namespace imageio */
#endif /* LSTLANDMARKFORMATPARSER_HPP_ */
