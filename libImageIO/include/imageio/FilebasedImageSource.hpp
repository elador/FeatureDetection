/*
 * FilebasedImageSource.hpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#ifndef FILEBASEDIMAGESOURCE_HPP_
#define FILEBASEDIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <vector>

using boost::filesystem::path;
using std::vector;

namespace imageio {

/**
 * Source of subsequent images, originating from files (that have a filename).
 */
class FilebasedImageSource : public ImageSource {
public:

	/**
	 * Get the path of the current image that is returned by the next
	 * call to ImageSource::get().
	 *
	 * @return The path to the image.
	 */
	virtual const path getPathOfNextImage() = 0;

	const vector<path> getPaths() { return files; }; // ...

protected:

	vector<path> files; ///< The files of the given directory, ordered by name.
	unsigned int index; ///< The index of the next file.
	// Todo Hmm where to initialize them now...  : files(), index(0)
};

} /* namespace imageio */
#endif /* FILEBASEDIMAGESOURCE_HPP_ */
