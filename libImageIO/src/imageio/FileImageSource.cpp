/*
 * FileImageSource.cpp
 *
 *  Created on: 26.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/FileImageSource.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdexcept>

using cv::imread;
using boost::filesystem::exists;
using std::sort;
using std::runtime_error;

namespace imageio {

FileImageSource::FileImageSource(string filePath) : files(), index(0) {
	path path(filePath);
	if (!exists(path))
		throw runtime_error("FileImageSource: File '" + filePath + "' does not exist.");
	/* TODO: Only copy valid images that opencv can handle. Those are:
		Built-in: bmp, portable image formats (pbm, pgm, ppm), Sun raster (sr, ras).
		With plugins, present by default: JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 (=Jasper)), 
										  TIFF files (tiff, tif), png.
		If specified: OpenEXR.
	*/
	files.push_back(path);
}

FileImageSource::FileImageSource(vector<string> filePaths) : files(), index(0) {

	/* TODO: Only copy valid images that opencv can handle. Those are:
		Built-in: bmp, portable image formats (pbm, pgm, ppm), Sun raster (sr, ras).
		With plugins, present by default: JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 (=Jasper)), 
										  TIFF files (tiff, tif), png.
		If specified: OpenEXR.
	*/
	for (auto file : filePaths) {
		path path(file);
		if (!exists(path))
			throw runtime_error("FileImageSource: File '" + file + "' does not exist.");
		files.push_back(path);
	}
	sort(files.begin(), files.end());
}

FileImageSource::~FileImageSource() {}

const Mat FileImageSource::get() {
	if (index >= files.size())
		return Mat();
	return imread(files[index++].string(), 1);
}

} /* namespace imageio */
