/*
 * ImageSink.h
 *
 *  Created on: 18.12.2012
 *      Author: poschmann
 */

#ifndef IMAGESINK_H_
#define IMAGESINK_H_

#include "opencv2/core/core.hpp"

using cv::Mat;

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
	virtual void add(const Mat& image) = 0;
};

} /* namespace imageio */
#endif /* IMAGESINK_H_ */
