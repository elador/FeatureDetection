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
class LandmarkSource;

/**
 * Image source that takes the images of an ImageSource and a LandmarkSource.
 *
 * Note: Maybe it doesn't make sense that LabeledImageSource inherits from FilebasedImageSource,
 *       because e.g. that means that it has to implement  its base-class function 
 *       getPathOfNextImage(), and that is conceptually a bit strange. (?) Only inheriting 
 *       from ImageSource doesn't make sense either because LabeledImageSource actually is a
 *       file-based source. Is there any downside of not inheriting from *ImageSource at all?
 *       What makes most sense?
 *       Maybe don't inherit but add a getImageSource() here, so we can still use it as an ImageSource?
 *
 *       It would maybe also be nice if we could just create an empty LandmarkSource at start, and then,
 *       at some time in the program, add the LandmarkSource from somewhere (e.g. user-specified).
 */
class LabeledImageSource : public ImageSource {
public:

	/**
	 * Constructs a new labeled image source.
	 *
	 * @param[in] imageSource The image source to use.
	 * @param[in] landmarkSource The landmark source to use.
	 */
	LabeledImageSource(shared_ptr<ImageSource> imageSource, shared_ptr<LandmarkSource> landmarkSource);

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

	const Mat getImage();

	const path getName();

	const vector<path> getNames();

	/**
	 * Retrieves the landmarks of the image that was last retrieved with get().
	 *
	 * @return The landmarks of the current image (that may be empty if no data could be retrieved).
	 */
	const LandmarkCollection getLandmarks();

private:

	shared_ptr<ImageSource> imageSource; ///< The underlying image source.
	shared_ptr<LandmarkSource> landmarkSource; ///< The underlying landmark source.

};

} /* namespace imageio */
#endif /* LABELEDIMAGESOURCE_HPP_ */
