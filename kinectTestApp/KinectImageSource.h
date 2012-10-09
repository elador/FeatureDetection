/*
 * KinectImageSource.h
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#ifndef VIDEOIMAGESOURCE_H_
#define VIDEOIMAGESOURCE_H_

#include "ImageSource.h"
#include "opencv2/highgui/highgui.hpp"
#include <NuiApi.h>

/**
 * Image source that takes images from a camera device or video file.
 */
class KinectImageSource : public ImageSource {
public:

	/**
	 * Constructs a new video image source that takes images from a camera device.
	 *
	 * @param[in] device ID of the video capturing device.
	 */
	explicit KinectImageSource(int device);


	virtual ~KinectImageSource();

	const cv::Mat get();

private:
	cv::VideoCapture capture; ///< The video capture.
	cv::Mat frame;            ///< The current frame.

	INuiSensor * m_pNuiSensor;	///< The kinect capture device.
	NUI_IMAGE_FRAME imageFrame; ///< The current frame.!

	HANDLE                  m_pColorStreamHandle; ///< The handle to the Kinect color stream we're grabbing from

};

#endif /* VIDEOIMAGESOURCE_H_ */
