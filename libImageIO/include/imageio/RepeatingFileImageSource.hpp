/*
 * RepeatingFileImageSource.hpp
 *
 *  Created on: 26.04.2013
 *      Author: Patrik Huber
 */

#ifndef REPEATINGFILEIMAGESOURCE_HPP_
#define REPEATINGFILEIMAGESOURCE_HPP_

#include "imageio/FilebasedImageSource.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"

using boost::filesystem::path;
using std::string;

namespace imageio {

/**
 * Image source that takes a single image and repeatedly outputs it.
 */
class RepeatingFileImageSource : public FilebasedImageSource {
public:

	/**
	 * Constructs a new repeating file image source.
	 *
	 * @param[in] file The path and filename of the image.
	 */
	RepeatingFileImageSource(string filePath);

	virtual ~RepeatingFileImageSource();

	const Mat get();

	const path getPathOfNextImage();

private:

	Mat image; ///< The image. As it's only one image, we pre-load it.
};

} /* namespace imageio */
#endif /* REPEATINGFILEIMAGESOURCE_HPP_ */
