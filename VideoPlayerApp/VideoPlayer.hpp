/*
 * VideoPlayer.hpp
 *
 *  Created on: 20.02.2014
 *      Author: poschmann
 */

#ifndef VIDEOPLAYER_HPP_
#define VIDEOPLAYER_HPP_

#include "imageio/ImageSource.hpp"
#include "imageio/ImageSink.hpp"
#include "imageio/AnnotationSource.hpp"
#include "opencv2/core/core.hpp"
#include <memory>
#include <vector>
#include <string>

class VideoPlayer {
public:

	VideoPlayer();

	void play(std::shared_ptr<imageio::ImageSource> imageSource,
			std::vector<std::shared_ptr<imageio::AnnotationSource>> annotationSources = std::vector<std::shared_ptr<imageio::AnnotationSource>>(),
			std::shared_ptr<imageio::ImageSink> imageSink = std::shared_ptr<imageio::ImageSink>());

private:

	static void strokeWidthChanged(int state, void* userdata);

	void drawAnnotation(cv::Mat& image, const cv::Rect& target, const cv::Scalar& color);

	static const std::string videoWindowName;
	static const std::string controlWindowName;

	bool paused;
	cv::Mat frame;
	cv::Mat image;
	std::vector<cv::Scalar> colors;
	float strokeWidth;
};

#endif /* VIDEOPLAYER_HPP_ */
