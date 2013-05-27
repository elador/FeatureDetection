/*
 * DefaultNamedLandmarkSource.hpp
 *
 *  Created on: 27.04.2013
 *      Author: Patrik Huber
 */

#ifndef DEFAULTNAMEDLANDMARKSOURCE_HPP_
#define DEFAULTNAMEDLANDMARKSOURCE_HPP_

#include "imageio/NamedLandmarkSource.hpp"
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
 * Named landmark source that reads from a list of landmark files using a parser.
 */
class DefaultNamedLandmarkSource : public NamedLandmarkSource {
public:

	/**
	 * Constructs a new landmark source from a list of landmark files and a
	 * parser to read the landmark files.
	 *
	 * @param[in] landmarkFiles A list of paths to landmark files.
	 * @param[in] fileParser A parser to read the file format.
	 */
	DefaultNamedLandmarkSource(vector<path> landmarkFiles, shared_ptr<LandmarkFormatParser> fileParser);
	
	~DefaultNamedLandmarkSource();

	const LandmarkCollection& get(const path& imagePath);

private:
	map<path, LandmarkCollection> landmarkCollections; ///< Holds all the landmarks for all images.
};

} /* namespace imageio */
#endif /* DEFAULTNAMEDLANDMARKSOURCE_HPP_ */
