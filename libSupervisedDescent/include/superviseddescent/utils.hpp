/*
 * utils.hpp
 *
 *  Created on: 29.07.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef SUPERVISEDDESCENT_UTILS_HPP_
#define SUPERVISEDDESCENT_UTILS_HPP_

#include "opencv2/core/core.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif

#include <string>

namespace superviseddescent {

/**
 * Free functions / classes belonging to the regression. (could also name the file linearalgebra.hpp? regression.hpp?)
 *
 */

// public?
// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
// - makes a copy of mean, not inplace
// - optional: scaling/trans that gets added to the mean (before scaling up to the facebox)
// Todo/Note: Is this the same as in SdmModel::alignRigid?
cv::Mat alignMean(cv::Mat mean, cv::Rect faceBox, float scalingX = 1.0f, float scalingY = 1.0f, float translationX = 0.0f, float translationY = 0.0f);

// mean translation to go from gt to esti
// use for x and y separately, i.e. separate them first
float calculateMeanTranslation(cv::Mat groundtruth, cv::Mat estimate);

// calc scale ratio of the estimate w.r.t. the GT
// i.e. if the estimate is estimated larger than the GT, it will return > 1.0f
// use for x and y separately, i.e. separate them first
float calculateScaleRatio(cv::Mat groundtruth, cv::Mat estimate);

// deals with both row and col vecs. Assumes first half x, second y.
void saveShapeInstanceToMatlab(cv::Mat shapeInstance, std::string filename);

// todo doc.
void drawLandmarks(cv::Mat image, cv::Mat landmarks, cv::Scalar color = cv::Scalar(0.0, 255.0, 0.0));


} /* namespace superviseddescent */

#endif /* SUPERVISEDDESCENT_UTILS_HPP_ */
