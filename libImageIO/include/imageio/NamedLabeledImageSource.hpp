/*
 * NamedLabeledImageSource.hpp
 *
 *  Created on: 27.04.2013
 *      Author: Patrik Huber
 */

#ifndef NAMEDLABELEDIMAGESOURCE_HPP_
#define NAMEDLABELEDIMAGESOURCE_HPP_

#include "imageio/LabeledImageSource.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem.hpp"
#include <memory>

using boost::filesystem::path;
using std::shared_ptr;

namespace imageio {

class NamedLandmarkSource;

/**
 * Image source with associated landmarks that are retrieved using the image names.
 *
 * Note: It might be nice if we could just create an empty LandmarkSource at start, and then,
 *       at some time in the program, add the LandmarkSource from somewhere (e.g. user-specified).
 */
class NamedLabeledImageSource : public LabeledImageSource {
public:

	/**
	 * Constructs a new named labeled image source from a given ImageSource and NamedLandmarkSource.
	 *
	 * @param[in] imageSource The image source to use.
	 * @param[in] landmarkSource The landmark source to use.
	 */
	NamedLabeledImageSource(shared_ptr<ImageSource> imageSource, shared_ptr<NamedLandmarkSource> landmarkSource);

	virtual ~NamedLabeledImageSource();

	const bool next();

	const Mat getImage() const;

	path getName() const;

	vector<path> getNames() const;

	const LandmarkCollection& getLandmarks() const;

private:
	shared_ptr<ImageSource> imageSource; ///< The underlying image source.
	shared_ptr<NamedLandmarkSource> landmarkSource; ///< The underlying landmark source.

};

} /* namespace imageio */
#endif /* NAMEDLABELEDIMAGESOURCE_HPP_ */
