/*
 * LabeledImageSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef LABELEDIMAGESOURCE_HPP_
#define LABELEDIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"

namespace imageio {

class LandmarkCollection;

/**
 * Image source with associated landmarks.
 */
class LabeledImageSource : public ImageSource {
public:

	virtual ~LabeledImageSource() {}

	virtual const bool next() = 0;

	virtual const Mat getImage() const = 0;

	virtual path getName() const = 0;

	virtual vector<path> getNames() const = 0;

	/**
	 * Retrieves the landmarks of the current image.
	 *
	 * @return The landmarks of the current image (that may be empty if no data could be retrieved).
	 */
	virtual const LandmarkCollection& getLandmarks() const = 0;
};

} /* namespace imageio */
#endif /* LABELEDIMAGESOURCE_HPP_ */
