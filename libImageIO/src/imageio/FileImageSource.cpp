/*
 * DirectoryImageSource.cpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#include "imageio/DirectoryImageSource.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using cv::imread;
using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;
using std::copy;
using std::sort;

namespace imageio {

DirectoryImageSource::DirectoryImageSource(string directory) : files(), index(0) {
	path path(directory);
	if (!exists(path))
		std::cerr << "directory '" << directory << "' does not exist" << std::endl;
	if (!is_directory(path))
		std::cerr << "'" << directory << "' is no directory" << std::endl;
	copy(directory_iterator(path), directory_iterator(), back_inserter(files));
	/* TODO: Only copy valid images that opencv can handle. Those are:
		Built-in: bmp, portable image formats (pbm, pgm, ppm), Sun raster (sr, ras).
		With plugins, present by default: JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 (=Jasper)), 
										  TIFF files (tiff, tif), png.
		If specified: OpenEXR.
	*/
	sort(files.begin(), files.end());
}

DirectoryImageSource::~DirectoryImageSource() {}

const Mat DirectoryImageSource::get() {
	if (index >= files.size())
		return Mat();
	return imread(files[index++].string(), 1);
}

} /* namespace imageio */
