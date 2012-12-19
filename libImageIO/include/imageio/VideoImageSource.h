/*
 * VideoImageSource.h
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#ifndef VIDEOIMAGESOURCE_H_
#define VIDEOIMAGESOURCE_H_

#include "ImageSource.h"
#include "opencv2/highgui/highgui.hpp"

using cv::VideoCapture;

namespace imageio {

/**
 * Image source that takes images from a camera device or video file.
 */
class VideoImageSource : public ImageSource {
public:

	/**
	 * Constructs a new video image source that takes images from a camera device.
	 *
	 * @param[in] device ID of the video capturing device.
	 */
	explicit VideoImageSource(int device);

	/**
	 * Constructs a new video image source that takes images from a video file.
	 *
	 * @param[in] video The name of the video file.
	 */
	explicit VideoImageSource(std::string video);

	virtual ~VideoImageSource();

	const Mat get();

private:
	VideoCapture capture; ///< The video capture.
	Mat frame;            ///< The current frame.
};

} /* namespace imageio */
#endif /* VIDEOIMAGESOURCE_H_ */
