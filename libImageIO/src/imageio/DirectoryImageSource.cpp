/*
 * DirectoryImageSource.cpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#include "imageio/DirectoryImageSource.h"
#include <iostream>

using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;

namespace imageio {

DirectoryImageSource::DirectoryImageSource(std::string directory) : files(), index(0) {
	path path(directory);
	if (!exists(path))
		std::cerr << "directory " << directory << " does not exist" << std::endl;
	if (!is_directory(path))
		std::cerr << directory << " is no directory" << std::endl;
	std::copy(directory_iterator(path), directory_iterator(), back_inserter(files));
	std::sort(files.begin(), files.end());
}

DirectoryImageSource::~DirectoryImageSource() {}

const cv::Mat DirectoryImageSource::get() {
	return cv::imread(files[index++].string(), 1);
}

} /* namespace imageio */
