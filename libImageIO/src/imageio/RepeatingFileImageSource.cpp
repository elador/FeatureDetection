/*
 * RepeatingFileImageSource.cpp
 *
 *  Created on: 26.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/RepeatingFileImageSource.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdexcept>

using cv::imread;
using boost::filesystem::exists;
using std::runtime_error;

namespace imageio {

RepeatingFileImageSource::RepeatingFileImageSource(string filePath) : ImageSource(filePath) {
	path file = path(filePath);
	if (!exists(file))
		throw runtime_error("RepeatingFileImageSource: File '" + filePath + "' does not exist.");

	/* TODO: Only load valid images that opencv can handle. Those are:
		Built-in: bmp, portable image formats (pbm, pgm, ppm), Sun raster (sr, ras).
		With plugins, present by default: JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 (=Jasper)), 
										  TIFF files (tiff, tif), png.
		If specified: OpenEXR.
	*/
	image = imread(file.string(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
		throw runtime_error("image '" + file.string() + "' could not be loaded");
	this->file = file;
}

RepeatingFileImageSource::~RepeatingFileImageSource() {}

void RepeatingFileImageSource::reset()
{
}

bool RepeatingFileImageSource::next()
{
	return true;
}

const Mat RepeatingFileImageSource::getImage() const
{
	return image;
}

path RepeatingFileImageSource::getName() const
{
	return file;
}

vector<path> RepeatingFileImageSource::getNames() const
{
	vector<path> tmp;
	tmp.push_back(file);
	return tmp;	// Todo: Figure out how to use initializer list here
}

} /* namespace imageio */
