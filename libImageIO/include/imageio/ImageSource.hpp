/*	
 * ImageSource.hpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#ifndef IMAGESOURCE_HPP_
#define IMAGESOURCE_HPP_

#include "opencv2/core/core.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <vector>

using cv::Mat;
using boost::filesystem::path;
using std::vector;

namespace imageio {

/**
 * Source of subsequent images.
 */
class ImageSource {
public:

	virtual ~ImageSource() {}

	/**
	 * Retrieves a single image.
	 *
	 * @return The image (that may be empty if no data could be retrieved).
	 */
	virtual const Mat get() = 0;

	/**
	 * Get the path of the current image that is returned by the next
	 * call to ImageSource::get().
	 *
	 * @return The path to the image.
	 */
	//virtual const path getPathOfNextImage() = 0;

	//const vector<path> getPaths() { return files; }; // ...

};

} /* namespace imageio */
#endif /* IMAGESOURCE_HPP_ */
