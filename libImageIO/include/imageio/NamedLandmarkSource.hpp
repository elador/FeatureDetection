/*
 * NamedLandmarkSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef NAMEDLANDMARKSOURCE_HPP_
#define NAMEDLANDMARKSOURCE_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"

namespace imageio {

class LandmarkCollection;

/**
 * Source of named landmark collections. Each landmark collection is named after the image
 * it belongs to.
 */
class NamedLandmarkSource {
public:

	virtual ~NamedLandmarkSource() {}

	/**
	* Resets the landmark source to its initial state.
	*/
	virtual void reset() = 0;

	/**
	* Moves the landmark source forward to the next collection of landmarks.
	*
	* @return True if the landmark source contains a next collection of landmarks, false otherwise.
	*/
	virtual bool next() = 0;

	/**
	* Retrieves the current collection of landmarks and moves the landmark source forward to
	* the next collection.
	*
	* @return The collection of landmarks (that may be empty if no data could be retrieved).
	*/
	virtual LandmarkCollection get() = 0;

	/**
	 * Retrieves the labels for a given image name.
	 *
	 * Performs a matching of the full path first, then, if
	 * no match is found, tries to match the basename only.
	 *
	 * @param imagePatch The full patch name of the image.
	 * @return The landmarks (that may be empty if no data could be retrieved).
	 */
	virtual LandmarkCollection get(const boost::filesystem::path& imagePath) = 0; // Note: This could be either implemented as modifying the state of the instance (iterator) or not. Decide and document! (see also DefaultNamedLandmarkSource - atm it doesn't change the iterator, so the function could in theory be const)

	/**
	* Retrieves the current collection of landmarks.
	* (Throws if iterator invalid. Maybe change, behaviour of other *Sources seems to be that it returns empty stuff if no data could be retrieved.)
	*
	* @return The collection of landmarks.
	*/
	virtual LandmarkCollection getLandmarks() const = 0;

	/**
	* Retrieves the name of the current collection of landmarks.
	* (Throws if iterator invalid. Maybe change, behaviour of other *Sources seems to be that it returns empty stuff if no data could be retrieved.)
	*
	* @return The name to which the current collection of landmarks belongs to.
	*/
	virtual boost::filesystem::path getName() const = 0;
};

} /* namespace imageio */
#endif /* NAMEDLANDMARKSOURCE_HPP_ */
