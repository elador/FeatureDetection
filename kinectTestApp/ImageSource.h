/*
 * ImageSource.h
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#ifndef IMAGESOURCE_H_
#define IMAGESOURCE_H_

#include "opencv2/highgui/highgui.hpp"

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
	virtual const cv::Mat get() = 0;
};

#endif /* IMAGESOURCE_H_ */
