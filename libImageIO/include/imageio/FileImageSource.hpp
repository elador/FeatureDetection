/*
 * FileImageSource.hpp
 *
 *  Created on: 24.04.2013
 *      Author: Patrik Huber
 */

#ifndef FILEIMAGESOURCE_HPP_
#define FILEIMAGESOURCE_HPP_

#include "ImageSource.hpp"
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
class FileImageSource : public ImageSource {
public:

	/**
	 * Constructs a new directory image source.
	 *
	 * @param[in] directory The directory containing image files.
	 */
	FileImageSource(string directory);

	virtual ~FileImageSource();

	const Mat get();

private:

	vector<path> files; ///< The files of the given directory, ordered by name.
	unsigned int index; ///< The index of the next file.
};

} /* namespace imageio */
#endif /* FILEIMAGESOURCE_HPP_ */
