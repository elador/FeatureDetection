/*
 * CameraImageSource.hpp
 *
 *  Created on: 27.09.2013
 *      Author: poschmann
 */

#ifndef CAMERAIMAGESOURCE_HPP_
#define CAMERAIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace imageio {

/**
 * Image source that takes images from a camera device.
 */
class CameraImageSource : public ImageSource {
public:

	/**
	 * Constructs a new camera image source.
	 *
	 * @param[in] device ID of the video capturing device.
	 */
	explicit CameraImageSource(int device);

	virtual ~CameraImageSource();

	void reset();

	bool next();

	const cv::Mat getImage() const;

	boost::filesystem::path getName() const;

	std::vector<boost::filesystem::path> getNames() const;

private:

	int device;           ///< ID of the video capturing device.
	cv::VideoCapture capture; ///< The video capture.
	cv::Mat frame;            ///< The current frame.
	unsigned long frameCounter; ///< The current frame number since the capture was started.
};

} /* namespace imageio */
#endif /* CAMERAIMAGESOURCE_HPP_ */
