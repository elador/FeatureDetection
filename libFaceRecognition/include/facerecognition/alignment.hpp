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
#include <array>

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


// Find a Delaunay triangulation of given points. Return the triangles.
// Generalisation of the OpenCV Delaunay example
// In: Points
// Out: Triangle list, with indices corresponding to the input points.
std::vector<std::array<int, 3>> delaunayTriangulate(std::vector<cv::Point2f> points);

/**
 * TODO
 *
 * From SDM Zhenhua, trained on PaSC, 5 semi-automatic landmarks.
 *
 */
class FivePointModel
{
public:
	FivePointModel();

	cv::Mat extractTexture2D(cv::Mat image, std::vector<cv::Point2f> landmarkPoints);

private:
	std::vector<cv::Point2f> points;
	std::vector<std::array<int, 3>> triangleList;
};

// Expects the given 5 points, in the following order:
// re_c, le_c, mouth_c, nt, botn
// Belongs to FivePointModel
std::vector<cv::Point2f> addArtificialPoints(std::vector<cv::Point2f> points);

// NOTE: Exact copy from libRender!
// Returns true if inside the tri or on the border
bool isPointInTriangle(cv::Point2f point, cv::Point2f triV0, cv::Point2f triV1, cv::Point2f triV2);

} /* namespace facerecognition */

#endif /* ALIGNMENT_HPP_ */
