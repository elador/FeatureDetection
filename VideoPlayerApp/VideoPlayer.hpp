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
#include "imageio/LandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "opencv2/core/core.hpp"
#include <memory>
#include <vector>
#include <string>

class VideoPlayer {
public:

	VideoPlayer();

	void play(std::shared_ptr<imageio::ImageSource> imageSource,
			std::vector<std::shared_ptr<imageio::LandmarkSource>> landmarkSources = std::vector<std::shared_ptr<imageio::LandmarkSource>>(),
			std::shared_ptr<imageio::ImageSink> imageSink = std::shared_ptr<imageio::ImageSink>());

private:

	static void strokeWidthChanged(int state, void* userdata);

	void drawLandmarks(cv::Mat& image, const imageio::LandmarkCollection& target, const cv::Scalar color);

	static const std::string videoWindowName;
	static const std::string controlWindowName;

	bool paused;
	cv::Mat frame;
	cv::Mat image;
	std::vector<cv::Scalar> colors;
	float strokeWidth;
};

#endif /* VIDEOPLAYER_HPP_ */
