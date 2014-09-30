/*	
 * ImageSource.hpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann, Patrik Huber
 */

#ifndef IMAGESOURCE_HPP_
#define IMAGESOURCE_HPP_

#include "opencv2/core/core.hpp"

namespace imageio {

/**
 * Source of subsequent images.
 */
class ImageSource {
public:

	virtual ~ImageSource() {}

	/**
	 * Retrieves the next image. The result is the same as calling next(), followed by getImage().
	 *
	 * @return The image (that may be empty if no data could be retrieved).
	 */
	const cv::Mat get() {
		if (!next())
			return cv::Mat();
		return getImage();
	}

	/**
	 * Resets this source to its initial state.
	 */
	virtual void reset() = 0;

	/**
	 * Proceeds to the next image of this source.
	 *
	 * @return True if successful (so there was another image), false otherwise.
	 */
	virtual bool next() = 0;

	/**
	 * Retrieves the current image.
	 *
	 * @return The image (that may be empty if no data could be retrieved).
	 */
	virtual const cv::Mat getImage() const = 0;
};

} /* namespace imageio */
#endif /* IMAGESOURCE_HPP_ */
