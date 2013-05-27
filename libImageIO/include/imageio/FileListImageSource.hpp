/*
 * FileListImageSource.hpp
 *
 *  Created on: 24.04.2013
 *      Author: Patrik Huber
 */

#ifndef FILELISTIMAGESOURCE_HPP_
#define FILELISTIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include "opencv2/core/core.hpp"
#include <vector>

using boost::filesystem::path;
using std::vector;
using std::string;
using cv::Mat;

namespace imageio {

/**
 * Image source that reads a text file containing an image filename on each line
 * and creates and image source from it.
 */
class FileListImageSource : public ImageSource {
public:

	/**
	 * Constructs a new file-list image source.
	 *
	 * @param[in] filelist A text-file containing a list of files.
	 */
	explicit FileListImageSource(const string& filelist);

	~FileListImageSource();

	const bool next();

	const Mat getImage() const;

	path getName() const;

	vector<path> getNames() const;

private:
	vector<path> files; ///< The files of the given directory, ordered by name.
	unsigned int index; ///< The index of the next file.
};

} /* namespace imageio */
#endif /* FILELISTIMAGESOURCE_HPP_ */
