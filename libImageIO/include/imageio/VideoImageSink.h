/*
 * VideoImageSink.h
 *
 *  Created on: 18.12.2012
 *      Author: poschmann
 */

#ifndef VIDEOIMAGESINK_H_
#define VIDEOIMAGESINK_H_

#include "imageio/ImageSink.h"
#include "opencv2/highgui/highgui.hpp"

using cv::VideoWriter;
using std::string;

namespace imageio {

/**
 * Image sink that stores images into a video file.
 */
class VideoImageSink : public ImageSink {
public:

	/**
	 * Constructs a new video image sink.
	 *
	 * @param[in] filename The name of the video file.
	 * @param[in] fps Framerate of the video stream.
	 * @param[in] fourcc 4-character code of video codec.
	 */
	explicit VideoImageSink(const string filename, double fps, int fourcc = CV_FOURCC('M','J','P','G'));

	~VideoImageSink();

	void add(const Mat& image);

private:
	const string filename; ///< The name of the video file.
	double fps;            ///< Framerate of the video stream.
	int fourcc;            ///< 4-character code of video codec.
	VideoWriter writer;    ///< The video writer.
};

} /* namespace imageio */
#endif /* VIDEOIMAGESINK_H_ */
