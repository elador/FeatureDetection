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
 *
 * Note: We could move common stuff from Ordered/Named here.
 * Also, it seems bad practice to derive from ImageSource and then
 * contain one (we already are one)? 
 */
class LabeledImageSource : public ImageSource {
public:

	/**
	 * Constructs a new labeled image source.
	 *
	 * @param[in] name The name of this image source.
	 */
	LabeledImageSource(const std::string& name) : ImageSource(name) {}

	virtual ~LabeledImageSource() {}

	virtual void reset() = 0;

	virtual bool next() = 0;

	virtual const cv::Mat getImage() const = 0;

	virtual boost::filesystem::path getName() const = 0;

	virtual std::vector<boost::filesystem::path> getNames() const = 0;

	/**
	 * Retrieves the landmarks of the current image.
	 *
	 * @return The landmarks of the current image (that may be empty if no data could be retrieved).
	 */
	virtual const LandmarkCollection getLandmarks() const = 0;
};

} /* namespace imageio */
#endif /* LABELEDIMAGESOURCE_HPP_ */
