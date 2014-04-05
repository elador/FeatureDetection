/*
 * DirectoryImageSource.hpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#ifndef DIRECTORYIMAGESOURCE_HPP_
#define DIRECTORYIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include "opencv2/core/core.hpp"
#include <vector>

using boost::filesystem::path;
using cv::Mat;
using std::vector;
using std::string;

namespace imageio {

/**
 * Image source that takes the images of a directory.
 */
class DirectoryImageSource : public ImageSource {
public:

	/**
	 * Constructs a new directory image source.
	 *
	 * @param[in] directory The directory containing image files.
	 */
	explicit DirectoryImageSource(const string& directory);

	~DirectoryImageSource();

	void reset();

	bool next();

	const Mat getImage() const;

	path getName() const;

	vector<path> getNames() const;

private:
	vector<path> files; ///< The files of the given directory, ordered by name.
	int index;			///< The index of the next file.

};

} /* namespace imageio */
#endif /* DIRECTORYIMAGESOURCE_HPP_ */
