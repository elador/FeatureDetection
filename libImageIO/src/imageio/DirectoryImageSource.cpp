/*
 * DirectoryImageSource.cpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#include "imageio/DirectoryImageSource.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdexcept>

using cv::imread;
using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;
using std::copy;
using std::sort;
using std::runtime_error;

namespace imageio {

DirectoryImageSource::DirectoryImageSource(string directory) : files(), index(0) {
	path dirpath(directory);
	if (!exists(dirpath))
		throw runtime_error("DirectoryImageSource: Directory '" + directory + "' does not exist.");
	if (!is_directory(dirpath))
		throw runtime_error("DirectoryImageSource: '" + directory + "' is not a directory.");
	copy(directory_iterator(dirpath), directory_iterator(), back_inserter(files));
	/* TODO: Only copy valid images that opencv can handle. Those are:
		Built-in: bmp, portable image formats (pbm, pgm, ppm), Sun raster (sr, ras).
		With plugins, present by default: JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 (=Jasper)), 
										  TIFF files (tiff, tif), png.
		If specified: OpenEXR.
	*/
	sort(files.begin(), files.end());
}

DirectoryImageSource::~DirectoryImageSource() {}

const bool DirectoryImageSource::next()
{
	if (index >= files.size()) {
		index++;
		return true;
	}
	return false;
}

const Mat DirectoryImageSource::get() {
	if (index >= files.size())
		return Mat();
	return imread(files[index++].string(), 1);	// output the current image and then increment index
}

const Mat DirectoryImageSource::getImage() const
{
	if (index >= files.size())
		return Mat();
	return imread(files[index].string(), 1);
}

const path DirectoryImageSource::getName() const
{
	return files[index];
}

const vector<path> DirectoryImageSource::getNames() const
{
	return files;
}

} /* namespace imageio */
