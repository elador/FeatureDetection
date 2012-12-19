/*
 * DirectoryImageSource.h
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#ifndef DIRECTORYIMAGESOURCE_H_
#define DIRECTORYIMAGESOURCE_H_

#include "ImageSource.h"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <vector>

using boost::filesystem::path;
using std::vector;
using std::string;

namespace imageio {

/**
 * Image source that takes the images of a directory.
 */
class DirectoryImageSource : public ImageSource {
public:

	/**
	 * Constructs a new directory image source.
	 *
	 * @param[in] directory The directory containing image files.
	 */
	DirectoryImageSource(string directory);

	virtual ~DirectoryImageSource();

	const Mat get();

private:

	vector<path> files; ///< The files of the given directory, ordered by name.
	unsigned int index; ///< The index of the next file.
};

} /* namespace imageio */
#endif /* DIRECTORYIMAGESOURCE_H_ */
