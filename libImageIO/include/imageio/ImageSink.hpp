/*
 * ImageSink.hpp
 *
 *  Created on: 18.12.2012
 *      Author: poschmann
 */

#ifndef IMAGESINK_HPP_
#define IMAGESINK_HPP_

#include "opencv2/core/core.hpp"

namespace imageio {

/**
 * Sink for subsequent images.
 */
class ImageSink {
public:

	virtual ~ImageSink() {}

	/**
	 * Adds an image.
	 *
	 * @param[in] image The image.
	 */
	virtual void add(const cv::Mat& image) = 0;
};

} /* namespace imageio */
#endif /* IMAGESINK_HPP_ */
