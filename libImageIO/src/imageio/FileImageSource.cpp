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
using cv::Mat;
using boost::filesystem::exists;
using boost::filesystem::path;
using std::sort;
using std::string;
using std::vector;
using std::runtime_error;

namespace imageio {

FileImageSource::FileImageSource(const string& filePath) : ImageSource(filePath), files(), index(-1) {
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

FileImageSource::FileImageSource(vector<string> filePaths) : ImageSource(filePaths.empty() ? "empty" : filePaths.front()), files(), index(-1) {

	/* TODO: Only copy valid images that opencv can handle. Those are:
		Built-in: bmp, portable image formats (pbm, pgm, ppm), Sun raster (sr, ras).
		With plugins, present by default: JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 (=Jasper)), 
										  TIFF files (tiff, tif), png.
		If specified: OpenEXR.
		Note: No, maybe don't filter here. See emails (search for "FileListImageSource")
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

void FileImageSource::reset()
{
	index = -1;
}

bool FileImageSource::next()
{
	index++;
	return index < static_cast<int>(files.size());
}

const Mat FileImageSource::getImage() const
{
	if (index < 0 || index >= static_cast<int>(files.size()))
		return Mat();
	Mat image = imread(files[index].string(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
		throw runtime_error("image '" + files[index].string() + "' could not be loaded");
	return image;
}

path FileImageSource::getName() const
{
	if (index < 0 || index >= static_cast<int>(files.size()))
		return path();
	return files[index];
}

vector<path> FileImageSource::getNames() const
{
	return files;
}

} /* namespace imageio */
