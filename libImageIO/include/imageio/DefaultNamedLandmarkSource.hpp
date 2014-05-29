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
#include "boost/filesystem/path.hpp"
#include <map>
#include <vector>
#include <memory>

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
	DefaultNamedLandmarkSource(std::vector<boost::filesystem::path> landmarkFiles, std::shared_ptr<LandmarkFormatParser> fileParser);
	
	void reset();

	bool next();

	// looks up using the given path (potentially full path) first, if not found, uses the basename
	// throws std::out_of_range exception if full path + basename both not found
	LandmarkCollection get(const boost::filesystem::path& imagePath); // This doesn't change the iterator at the moment, it's completely independent of it.

	LandmarkCollection getLandmarks() const;

	boost::filesystem::path getName() const;

private:
	std::map<boost::filesystem::path, LandmarkCollection> landmarkCollections;		///< Holds all the landmarks for all images.
	std::map<boost::filesystem::path, LandmarkCollection>::const_iterator index;	///< The current landmark position in the landmarkCollections map.
	bool iteratorIsBeforeBegin;	///< Specifies whether we have started iterating through the landmarks yet. This is a little hack because we can't set index to -1 so that it points to the first valid element after one call to next().
};

} /* namespace imageio */
#endif /* DEFAULTNAMEDLANDMARKSOURCE_HPP_ */
