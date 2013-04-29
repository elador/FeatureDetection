/*
 * LandmarkFormatParser.hpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#ifndef LANDMARKFORMATPARSER_HPP_
#define LANDMARKFORMATPARSER_HPP_

#include "imageio/LandmarkCollection.hpp"
#include <vector>
#include <string>

using std::vector;
using std::string;

namespace imageio {

/**
 * Takes one or several lines from a landmarks file for one image as 
 * input and returns a LandmarkCollection (TODO in tlms format) with all the landmarks found.
 */
class LandmarkFormatParser {
public:

	virtual ~LandmarkFormatParser() {}

	/**
	 * Reads the landmark data for one single image and returns all its landmarks (TODO in tlms format).
	 *
	 * Note: The input could also be XML (a tree node or something). Use-case for boost::optional?
	 *       Or maybe don't use inheritance here, but several read-functions, e.g. readTlms(vec<str>), 
	 *       readRavlXml(xml_node), ... ? But that would make LandmarkFileLoader more complicated because
	 *       it couldn't just call one read(...) ?
	 *       Or maybe just use one LandmarkFormatParser class with an optional and an enum TYPE and some if's...
	 *       I think separate classes are quite good because they need quite a lot of helper routines/structures
	 *       and that gets too messy if it's all in one class!
	 *       But actually passing a vector<string> is really not so good, it's not intuitive, what if a file
	 *       is not read line-by-line, etc...
	 *
	 * @param[in] landmarkData One or several lines from a landmarks file for one image.
	 * @return All the landmarks that are present in the input (TODO in tlms format).
	 */
	virtual const LandmarkCollection read(vector<string> landmarkData) = 0;
};

} /* namespace imageio */
#endif /* LANDMARKFORMATPARSER_HPP_ */
