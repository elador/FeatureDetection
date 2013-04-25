/*
 * FileImageSource.hpp
 *
 *  Created on: 24.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/FileImageSource.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using cv::imread;
using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;
using std::copy;
using std::sort;

namespace imageio {

FileImageSource::FileImageSource(string directory) : files(), index(0) {
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

FileImageSource::~FileImageSource() {}

const Mat FileImageSource::get() {
	if (index >= files.size())
		return Mat();
	return imread(files[index++].string(), 1);
}

} /* namespace imageio */
