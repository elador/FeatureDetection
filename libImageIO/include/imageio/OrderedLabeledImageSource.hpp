/*
 * OrderedLabeledImageSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef ORDEREDLABELEDIMAGESOURCE_HPP_
#define ORDEREDLABELEDIMAGESOURCE_HPP_

#include "imageio/LabeledImageSource.hpp"
#include <memory>

using std::shared_ptr;

namespace imageio {

class OrderedLandmarkSource;

/**
 * Labeled image source whose landmarks are associated to the images by their order.
 */
class OrderedLabeledImageSource : public LabeledImageSource {
public:

	/**
	 * Constructs a new ordered labeled image source.
	 *
	 * @param[in] imageSource The source of the images.
	 * @param[in] landmarkSource The source of the landmarks.
	 */
	OrderedLabeledImageSource(shared_ptr<ImageSource> imageSource, shared_ptr<OrderedLandmarkSource> landmarkSource);

	~OrderedLabeledImageSource();

	const bool next();

	const Mat getImage() const;

	path getName() const;

	vector<path> getNames() const;

	const LandmarkCollection& getLandmarks() const;

private:

	shared_ptr<ImageSource> imageSource;              ///< The source of the images.
	shared_ptr<OrderedLandmarkSource> landmarkSource; ///< The source of the landmarks.
};

} /* namespace imageio */
#endif /* ORDEREDLABELEDIMAGESOURCE_HPP_ */
