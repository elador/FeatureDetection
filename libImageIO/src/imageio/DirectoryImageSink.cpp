/*
 * DirectoryImageSink.cpp
 *
 *  Created on: 18.12.2012
 *      Author: poschmann
 */

#include "imageio/DirectoryImageSink.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <sstream>
#include <iostream>
#include <iomanip>

using cv::Mat;
using boost::filesystem::path;
using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::create_directory;
using std::string;
using std::ostringstream;
using std::setfill;
using std::setw;

namespace imageio {

DirectoryImageSink::DirectoryImageSink(string directory, string ending) :
		directory(directory), ending(ending), index(0) {
	if (this->directory[directory.length() - 1] != '/')
		this->directory += '/';
	path path(this->directory);
	if (!exists(path)) {
		if (!create_directory(path))
			std::cerr << "Could not create directory '" << directory << "'" << std::endl;
	} else if (!is_directory(path))
		std::cerr << "'" << directory << "' is no directory" << std::endl;
}

DirectoryImageSink::~DirectoryImageSink() {}

void DirectoryImageSink::add(const Mat& image) {
	ostringstream filename;
	filename << directory << setfill('0') << setw(5) << index++ << setw(0) << '.' << ending;
	if (!imwrite(filename.str(), image))
		std::cerr << "Could not write image file '" << filename.str() << "'" << std::endl;
}

} /* namespace imageio */
