/*
 * KinectImageSource.h
 *
 *  Created on: 07.10.2012
 *      Author: Patrik Huber
 */

#ifndef KINECTIMAGESOURCE_H_
#define KINECTIMAGESOURCE_H_

#include "ImageSource.h"
#ifdef WIN32
	#include <windows.h>
	#include <NuiApi.h>
#endif

namespace imageio {

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

	const Mat get();

private:
	Mat frame; ///< The current frame.

#ifdef WIN32
	INuiSensor * m_pNuiSensor;	///< The kinect capture device.
	NUI_IMAGE_FRAME imageFrame; ///< The current frame.!

	HANDLE                  m_pColorStreamHandle; ///< The handle to the Kinect color stream we're grabbing from
#endif

};

} /* namespace imageio */

#endif /* KINECTIMAGESOURCE_H_ */
