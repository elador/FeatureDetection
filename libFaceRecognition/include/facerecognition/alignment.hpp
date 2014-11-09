/*
 * alignment.hpp
 *
 *  Created on: 09.11.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef ALIGNMENT_HPP_
#define ALIGNMENT_HPP_

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"
#include "boost/optional.hpp"

#include "opencv2/core/core.hpp"

#include <vector>

namespace facerecognition {

// rigid rotation
// getInPlaneRotation? (get2DRot...?)
// assumes second pair of eyes is horizontal, aligning to that
// right eye is right, as in a 3D model
cv::Mat getRotationMatrixFromEyePairs(cv::Vec2f rightEye, cv::Vec2f leftEye);

// refs: optional out params, to know what happened. I.e. if we want to translate all landmarks.
// Crop widthFactor * ied to the left and right, i.e. total width will be 2 * widthFactor * ied
// Crop heightFactor * ied to the top, and 2 * heightFactor * ied to the bottom
// Crops & aligns in original image, i.e. output also original (maximum possible) resolution
// What happens if the cropping goes outside the borders? Atm OpenCV throws.
// We could throw a custom exception, or check and use copyMakeBorder?
cv::Mat cropAligned(cv::Mat image, cv::Vec2f rightEye, cv::Vec2f leftEye, float widthFactor=1.1f, float heightFactor=0.8f, float* translationX=nullptr, float* translationY=nullptr);

} /* namespace facerecognition */

#endif /* ALIGNMENT_HPP_ */
