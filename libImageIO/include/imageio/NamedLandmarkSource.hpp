/*
 * NamedLandmarkSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef NAMEDLANDMARKSOURCE_HPP_
#define NAMEDLANDMARKSOURCE_HPP_

#include "boost/filesystem.hpp"

using boost::filesystem::path;

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
	 * Retrieves the labels for a given image name.
	 *
	 * Performs a matching of the full path first, then, if
	 * no match is found, tries to match the basename only.
	 *
	 * @param imagePatch The full patch name of the image.
	 * @return The landmarks (that may be empty if no data could be retrieved).
	 */
	virtual const LandmarkCollection get(path imagePath) = 0;
};

} /* namespace imageio */
#endif /* NAMEDLANDMARKSOURCE_HPP_ */
