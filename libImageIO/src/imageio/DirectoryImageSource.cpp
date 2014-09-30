/*
 * DirectoryImageSource.cpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#include "imageio/DirectoryImageSource.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <stdexcept>

using cv::Mat;
using cv::imread;
using boost::filesystem::path;
using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;
using std::string;
using std::vector;
using std::runtime_error;

namespace imageio {

DirectoryImageSource::DirectoryImageSource(const string& directory) : files(), index(-1) {
	path dirpath(directory);
	if (!exists(dirpath))
		throw runtime_error("DirectoryImageSource: Directory '" + directory + "' does not exist.");
	if (!is_directory(dirpath))
		throw runtime_error("DirectoryImageSource: '" + directory + "' is not a directory.");

	vector<path> allFiles;
	std::copy(directory_iterator(dirpath), directory_iterator(), back_inserter(allFiles));
	vector<string> imageExtensions = { "bmp", "dib", "pbm", "pgm", "ppm", "sr", "ras", "jpeg", "jpg", "jpe", "jp2", "png", "tiff", "tif" };
	auto newFilesEnd = std::remove_if(allFiles.begin(), allFiles.end(), [&](const path& file) {
		string extension = file.extension().string();
		if (extension.size() > 0)
			extension = extension.substr(1);
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
		return std::none_of(imageExtensions.begin(), imageExtensions.end(), [&](const string& imageExtension) {
			return imageExtension == extension;
		});
	});
	allFiles.erase(newFilesEnd, allFiles.end());
	std::sort(allFiles.begin(), allFiles.end());
	files.resize(allFiles.size());
	std::transform(allFiles.begin(), allFiles.end(), files.begin(), [](const path& file) {
		return file.string();
	});
}

void DirectoryImageSource::reset() {
	index = -1;
}

bool DirectoryImageSource::next() {
	index++;
	return index < static_cast<int>(files.size());
}

const Mat DirectoryImageSource::getImage() const {
	if (index < 0 || index >= static_cast<int>(files.size()))
		return Mat();
	return imread(files[index], CV_LOAD_IMAGE_COLOR);
}

} /* namespace imageio */
