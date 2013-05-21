/*
 * DefaultLandmarkSource.hpp
 *
 *  Created on: 27.04.2013
 *      Author: Patrik Huber
 */

#ifndef DEFAULTLANDMARKSOURCE_HPP_
#define DEFAULTLANDMARKSOURCE_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <map>
#include <vector>
#include <memory>

using boost::filesystem::path;
using std::map;
using std::vector;
using std::shared_ptr;

namespace imageio {

class LandmarkCollection;
class LandmarkFormatParser;

/**
 * Default landmark source taking a list of landmark files and a parser for those files.
 */
class DefaultLandmarkSource {
public:

	/**
	 * Constructs a new landmark source from a list of landmark files and a
	 * parser to read the landmark files.
	 *
	 * @param[in] landmarkFiles A list of paths to landmark files.
	 * @param[in] fileParser A parser to read the file format.
	 */
	DefaultLandmarkSource(vector<path> landmarkFiles, shared_ptr<LandmarkFormatParser> fileParser);
	
	~DefaultLandmarkSource();

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
#endif /* DEFAULTLANDMARKSOURCE_HPP_ */
