/*
 * FileListImageSource.cpp
 *
 *  Created on: 24.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/FileListImageSource.hpp"
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

FileListImageSource::FileListImageSource(string filelist) : files(), index(0) {
	path path(filelist);
	if (!exists(path))
		throw runtime_error("File '" + filelist + "' does not exist");

	std::ifstream listfileStream;
	listfileStream.open(filelist.c_str(), std::ios::in);
	if (!listfileStream.is_open()) {
		throw runtime_error("FileListImageSource: Error opening file list!");
	}
	string line;
	while (listfileStream.good()) {
		getline(listfileStream, line);
		if(line=="") {
			continue;
		}
		string buf;
		std::stringstream ss(line);
		ss >> buf;	
		files.push_back(boost::filesystem::path(buf));	// Insert the image filename, just ignore the rest of the line
	}
	listfileStream.close();

	/* TODO: Only copy valid images that opencv can handle. Those are:
		Built-in: bmp, portable image formats (pbm, pgm, ppm), Sun raster (sr, ras).
		With plugins, present by default: JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 (=Jasper)), 
										  TIFF files (tiff, tif), png.
		If specified: OpenEXR.
	*/
	// sort(files.begin(), files.end()); // Note: I think we do not want to sort the files
										 //       if they come from a list.
}

FileListImageSource::~FileListImageSource() {}

const bool FileListImageSource::next()
{
	if (index >= files.size()) {
		index++;
		return true;
	}
	return false;
}

const Mat FileListImageSource::get() {
	if (index >= files.size())
		return Mat();
	return imread(files[index++].string(), 1);	// output the current image and then increment index
}

const Mat FileListImageSource::getImage()
{
	if (index >= files.size())
		return Mat();
	return imread(files[index].string(), 1);
}

const path FileListImageSource::getName()
{
	return files[index];
}

const vector<path> FileListImageSource::getNames()
{
	return files;
}

} /* namespace imageio */
