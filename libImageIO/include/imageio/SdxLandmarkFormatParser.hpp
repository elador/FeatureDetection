/*
 * SdxLandmarkFormatParser.hpp
 *
 *  Created on: 05.08.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef SDXLANDMARKFORMATPARSER_HPP_
#define SDXLANDMARKFORMATPARSER_HPP_

#include "imageio/LandmarkFormatParser.hpp"
#include "imageio/LandmarkCollection.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"
#include <map>

namespace imageio {

/**
 * Reads landmarks from a text file in the .sdx file format provided
 * by Cognitec. The file contains one header line. The format looks
 * like the following:
 *
 * filename LeftEyeX LeftEyeY RightEyeX RightEyeY MouthX MouthY NoseTipX NoseTipY BridgeOfTheNoseX BridgeOfTheNoseY
 * example.jpg 1592 1501 1751 1510 1669.42 1672.41 1683.38 1568.25 1678.44 1497.51
 * ...
 *
 */
class SdxLandmarkFormatParser : public LandmarkFormatParser {
public:

	/**
	 * Reads the landmark data for one single image and returns all its landmarks.
	 *
	 * @param[in] landmarkFilePath A path to a MUCT .csv file.
	 * @return A map with an entry for every image and every landmark.
	 */
	const std::map<boost::filesystem::path, LandmarkCollection> read(boost::filesystem::path landmarkFilePath);

};

} /* namespace imageio */
#endif /* SDXLANDMARKFORMATPARSER_HPP_ */
