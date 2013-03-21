/*
 * ImageSource.hpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#ifndef IMAGESOURCE_HPP_
#define IMAGESOURCE_HPP_

#include "opencv2/core/core.hpp"

using cv::Mat;

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
};

} /* namespace imageio */
#endif /* IMAGESOURCE_HPP_ */
