/*
 * VideoImageSource.hpp
 *
 *  Created on: 20.08.2012
 *      Author: poschmann
 */

#ifndef VIDEOIMAGESOURCE_HPP_
#define VIDEOIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"
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

	const bool next();

	const Mat getImage() const;

	path getName() const;

	vector<path> getNames() const;

private:
	VideoCapture capture; ///< The video capture.
	Mat frame;            ///< The current frame.
	unsigned long frameCounter; ///< The current frame number since the capture was started.
};

} /* namespace imageio */
#endif /* VIDEOIMAGESOURCE_HPP_ */
