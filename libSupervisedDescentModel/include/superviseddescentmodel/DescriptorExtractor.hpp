/*
 * DescriptorExtractor.hpp
 *
 *  Created on: 21.03.2014
 *      Author: Patrik Huber
 */

#pragma once

#ifndef DESCRIPTOREXTRACTOR_HPP_
#define DESCRIPTOREXTRACTOR_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif

extern "C" {
	#include "superviseddescentmodel/hog.h"
}

using cv::Mat;
using std::vector;


namespace superviseddescentmodel {

/**
 * Base-class for extracting descriptors at given points
 * in an image.
 * Todo: Check how this integrates with our FeatureExtractors
 * in libImageProcessing.
 */
class DescriptorExtractor
{
public:
	// returns a Matrix, as many rows as points, 1 descriptor = 1 row
	virtual cv::Mat getDescriptors(const cv::Mat image, std::vector<cv::Point2f> locations) = 0;
};

class SiftDescriptorExtractor : public DescriptorExtractor
{
public:
	// c'tor with param diameter & orientation? (0.0f = right?)
	// and store them as private vars.
	// However, it might be better to store the parameters separately, to be able to share a FeatureDescriptorExtractor over multiple Sdm cascade levels

	cv::Mat getDescriptors(const cv::Mat image, std::vector<cv::Point2f> locations) {
		Mat grayImage;
		if (image.channels() == 3) {
			cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
		} else {
			grayImage = image;
		}
		cv::SIFT sift; // init only once, could reuse
		vector<cv::KeyPoint> keypoints;
		for (const auto& loc : locations) {
			keypoints.emplace_back(cv::KeyPoint(loc, 32.0f, 0.0f)); // Angle is set to 0. If it's -1, SIFT will be calculated for 361degrees. But Paper (email) says upwards.
		}
		Mat siftDescriptors;
		sift(grayImage, Mat(), keypoints, siftDescriptors, true);
		//cv::drawKeypoints(img, keypoints, img, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		return siftDescriptors;
	};
};

class HogDescriptorExtractor : public DescriptorExtractor
{
public:
	// Store the params as private vars? However, it might be better to store the parameters separately, to be able to share a FeatureDescriptorExtractor over multiple Sdm cascade levels

	cv::Mat getDescriptors(const cv::Mat image, std::vector<cv::Point2f> locations) {
		Mat grayImage;
		if (image.channels() == 3) {
			cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
		}
		else {
			grayImage = image;
		}
		
		// do it

		Mat hogDescriptors;
		return hogDescriptors;
	};
};


} /* namespace superviseddescentmodel */
#endif /* DESCRIPTOREXTRACTOR_HPP_ */
