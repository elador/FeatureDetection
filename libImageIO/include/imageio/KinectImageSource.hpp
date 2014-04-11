/*
 * KinectImageSource.hpp
 *
 *  Created on: 07.10.2012
 *      Author: Patrik Huber
 */

#ifndef KINECTIMAGESOURCE_HPP_
#define KINECTIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"
#ifdef WITH_MSKINECT_SDK
	#define NOMINMAX	// This specifies that windows.h does not #define it's min/max macros.
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

	~KinectImageSource();

	void reset();

	bool next();

	const cv::Mat getImage() const;

	boost::filesystem::path getName() const;

	std::vector<boost::filesystem::path> getNames() const;

private:
	cv::Mat frame; ///< The current frame.

#ifdef WITH_MSKINECT_SDK
	INuiSensor * m_pNuiSensor;	///< The kinect capture device.
	NUI_IMAGE_FRAME imageFrame; ///< The current frame.
	HANDLE m_pColorStreamHandle; ///< The handle to the Kinect color stream we're grabbing from.
#endif

};

} /* namespace imageio */

#endif /* KINECTIMAGESOURCE_HPP_ */
