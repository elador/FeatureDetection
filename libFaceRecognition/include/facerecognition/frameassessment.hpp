/*
 * frameassessment.hpp
 *
 *  Created on: 08.11.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FRAMEASSESSMENT_HPP_
#define FRAMEASSESSMENT_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"

#include "opencv2/core/core.hpp"

#include <vector>

namespace facerecognition {

float sharpnessScoreCanny(cv::Mat frame);

// Should also try this one: http://stackoverflow.com/a/7767755/1345959
// Focus-measurement algorithms, from http://stackoverflow.com/a/7768918/1345959:

// OpenCV port of 'LAPM' algorithm (Nayar89)
double modifiedLaplacian(const cv::Mat& src);

// OpenCV port of 'LAPV' algorithm (Pech2000)
double varianceOfLaplacian(const cv::Mat& src);

// OpenCV port of 'TENG' algorithm (Krotkov86)
double tenengrad(const cv::Mat& src, int ksize);

// OpenCV port of 'GLVN' algorithm (Santos97)
double normalizedGraylevelVariance(const cv::Mat& src);

// Could all be renamed to "transform/fitLinear" or something
// Todo: Proper doc
// Note: Segfaults if the given vector is empty (iterator not dereferentiable)
std::vector<float> minMaxFitTransformLinear(std::vector<float> values);

std::vector<float> getVideoNormalizedYawPoseScores(std::vector<float> yaws);

} /* namespace facerecognition */

#endif /* FRAMEASSESSMENT_HPP_ */
