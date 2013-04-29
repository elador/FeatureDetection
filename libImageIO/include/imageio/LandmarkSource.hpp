/*
 * LandmarkSource.hpp
 *
 *  Created on: 27.04.2013
 *      Author: Patrik Huber
 */

#ifndef LANDMARKSOURCE_HPP_
#define LANDMARKSOURCE_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <map>
#include <memory>

using boost::filesystem::path;
using std::map;
using std::shared_ptr;

namespace imageio {

class LandmarkCollection;
class LandmarkFileLoader;

/**
 * Source of labeled image landmarks.
 */
class LandmarkSource {
public:

	/**
	 * Constructs a new landmark source.
	 *
	 * Todo: Maybe we want to take a shared_ptr<LandmarkFileLoader> landmarkFileLoader
	 *       but then LandmarkFileLoader gets more complicated.
	 *
	 * @param[in] directory The directory containing image files.
	 */
	LandmarkSource(map<path, LandmarkCollection> landmarkCollections);
	
	~LandmarkSource();

	/**
	 * Retrieves the labels for a given image.
	 *
	 * Performs a matching of the full path first, then, if
	 * no match is found, tries to match the basename only.
	 *
	 * @return The landmarks (that may be empty if no data could be retrieved).
	 */
	const LandmarkCollection get(path imagePath);

private:
	map<path, LandmarkCollection> landmarkCollections; ///< Holds all the landmarks for all images.
};

} /* namespace imageio */
#endif /* LANDMARKSOURCE_HPP_ */
