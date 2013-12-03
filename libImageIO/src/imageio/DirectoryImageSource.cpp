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

DirectoryImageSource::DirectoryImageSource(const string& directory) : ImageSource(directory), files(), index(-1) {
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
		Parameter for list of valid file extensions?
		Parameter for predicate (used by remove_if)?
		Parameter for single file extension?
		Unify the different image sources that work with file lists (DirectoryImageSource,
		FileImageSource, FileListImageSource, RepeatingFileImageSourcec), so the filtering
		does not have to be repeated in each of them. But not all of them need the filtering
		anyway (for example if a file list is given explicitly).

		First prototype version of file filtering by extension:
	*/
	vector<string> imageExtensions; // No initializer lists in VS2012 yet. Change back as soon as we're on VS2013.
	imageExtensions.push_back("bmp");
	imageExtensions.push_back("dib");
	imageExtensions.push_back("pbm");
	imageExtensions.push_back("pgm");
	imageExtensions.push_back("ppm");
	imageExtensions.push_back("sr");
	imageExtensions.push_back("ras");
	imageExtensions.push_back("jpeg");
	imageExtensions.push_back("jpg");
	imageExtensions.push_back("jpe");
	imageExtensions.push_back("jp2");
	imageExtensions.push_back("png");
	imageExtensions.push_back("tiff");
	imageExtensions.push_back("tif");
	auto newFilesEnd = std::remove_if(files.begin(), files.end(), [&](const path& file) {
		string extension = file.extension().string();
		if (extension.size() > 0)
			extension = extension.substr(1);
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
		return std::none_of(imageExtensions.begin(), imageExtensions.end(), [&](const string& imageExtension) {
			return imageExtension == extension;
		});
	});
	files.erase(newFilesEnd, files.end());

	sort(files.begin(), files.end());
}

DirectoryImageSource::~DirectoryImageSource() {}

void DirectoryImageSource::reset()
{
	index = -1;
}

bool DirectoryImageSource::next()
{
	index++;
	return index < files.size();
}

const Mat DirectoryImageSource::getImage() const
{
	if (index < 0 || index >= files.size())
		return Mat();
	return imread(files[index].string(), CV_LOAD_IMAGE_COLOR);
}

path DirectoryImageSource::getName() const
{
	if (index < 0 || index >= files.size())
		return path();
	return files[index];
}

vector<path> DirectoryImageSource::getNames() const
{
	return files;
}

} /* namespace imageio */
