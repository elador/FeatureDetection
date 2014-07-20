/*
 * OpenCVSiftFilter.cpp
 *
 *  Created on: 03.07.2014
 *      Author: Patrik Huber
 */

#include "superviseddescent/OpenCVSiftFilter.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include <vector>

using cv::Mat;

namespace superviseddescent {

Mat OpenCVSiftFilter::applyTo(const Mat& image, Mat& filtered) const
{
	//cv::Mat getDescriptors(const cv::Mat image, std::vector<cv::Point2f> locations) {
	Mat grayImage;
	if (image.channels() == 3) {
		cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
	}
	else {
		grayImage = image;
	}
	cv::SIFT sift; // init only once, could reuse
	std::vector<cv::KeyPoint> keypoints;
	//for (const auto& loc : locations) {
	//	keypoints.emplace_back(cv::KeyPoint(loc, 32.0f, 0.0f)); // Angle is set to 0. If it's -1, SIFT will be calculated for 361degrees. But Paper (email) says upwards.
	//}
	Mat siftDescriptors;
	sift(grayImage, Mat(), keypoints, siftDescriptors, true); // TODO: What happens if the keypoint (or part of its patch) is outside the image?
	//cv::drawKeypoints(img, keypoints, img, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	return siftDescriptors;
	//};
	
	filtered = Mat();
	return filtered;
}

} /* namespace superviseddescent */
