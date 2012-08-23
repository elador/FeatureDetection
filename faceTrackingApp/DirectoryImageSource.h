/*
 * DirectoryImageSource.h
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#ifndef DIRECTORYIMAGESOURCE_H_
#define DIRECTORYIMAGESOURCE_H_

#include "ImageSource.h"
#include "boost/filesystem.hpp"
#include <vector>

using boost::filesystem::path;

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
	DirectoryImageSource(std::string directory);

	virtual ~DirectoryImageSource();

	const cv::Mat get();

private:

	std::vector<path> files; ///< The files of the given directory, ordered by name.
	int index;               ///< The index of the next file.
};

#endif /* DIRECTORYIMAGESOURCE_H_ */
