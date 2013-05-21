/*
 * LabeledImageSource.hpp
 *
 *  Created on: 27.04.2013
 *      Author: Patrik Huber
 */

#ifndef LABELEDIMAGESOURCE_HPP_
#define LABELEDIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <memory>

using boost::filesystem::path;
using std::shared_ptr;

namespace imageio {

class LandmarkCollection;
class DefaultLandmarkSource;

/**
 * Image source that takes the images of an ImageSource and a LandmarkSource.
 *
 * Note: It might be nice if we could just create an empty LandmarkSource at start, and then,
 *       at some time in the program, add the LandmarkSource from somewhere (e.g. user-specified).
 */
class LabeledImageSource : public ImageSource {
public:

	/**
	 * Constructs a new labeled image source from a given ImageSource and LandmarkSource.
	 *
	 * @param[in] imageSource The image source to use.
	 * @param[in] landmarkSource The landmark source to use.
	 */
	LabeledImageSource(shared_ptr<ImageSource> imageSource, shared_ptr<DefaultLandmarkSource> landmarkSource);

	virtual ~LabeledImageSource();

	/**
	 * Retrieves the image of the underlying image source.
	 *
	 * In addition, saves the path of the retrieved image to be
	 * able to output the labels for it if getLandmarks() is called.
	 *
	 * @return The image (that may be empty if no data could be retrieved).
	 */
	const Mat get();

	const bool next();

	const Mat getImage() const;

	const path getName() const;

	const vector<path> getNames() const;

	/**
	 * Retrieves the landmarks of the image that was last retrieved with get().
	 *
	 * @return The landmarks of the current image (that may be empty if no data could be retrieved).
	 */
	const LandmarkCollection getLandmarks();

private:
	shared_ptr<ImageSource> imageSource; ///< The underlying image source.
	shared_ptr<DefaultLandmarkSource> landmarkSource; ///< The underlying landmark source.

};

} /* namespace imageio */
#endif /* LABELEDIMAGESOURCE_HPP_ */
