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

namespace imageio {

/**
 * Image source that takes images from a video file.
 */
class VideoImageSource : public ImageSource {
public:

	/**
	 * Constructs a new video image source.
	 *
	 * @param[in] video The name of the video file.
	 */
	explicit VideoImageSource(std::string video);

	virtual ~VideoImageSource();

	void reset();

	bool next();

	const cv::Mat getImage() const;

	boost::filesystem::path getName() const;

	std::vector<boost::filesystem::path> getNames() const;

private:

	std::string video;         ///< The name of the video file.
	cv::VideoCapture capture; ///< The video capture.
	cv::Mat frame;            ///< The current frame.
	unsigned long frameCounter; ///< The current frame number since the capture was started.
};

} /* namespace imageio */
#endif /* VIDEOIMAGESOURCE_HPP_ */
