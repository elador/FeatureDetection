/*
 * NamedLandmarkSink.hpp
 *
 *  Created on: 05.04.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef NAMEDLANDMARKSINK_HPP_
#define NAMEDLANDMARKSINK_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"

namespace imageio {

class LandmarkCollection;

/**
 * Sink for landmark collections where each landmark
 * is saved to a separate file with a filename.
 */
class NamedLandmarkSink {
public:

	virtual ~NamedLandmarkSink() {}

	/**
	 * Adds a landmark collection and saves it to the given file.
	 *
	 * @param[in] collection The landmark collection.
	 * @param[in] filename The file to which to save the landmarks, without file extension.
	 */
	virtual void add(const LandmarkCollection& collection, boost::filesystem::path filename) = 0;
};

} /* namespace imageio */
#endif /* NAMEDLANDMARKSINK_HPP_ */
