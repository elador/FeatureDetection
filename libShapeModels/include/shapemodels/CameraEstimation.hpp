/*
 * CameraEstimation.hpp
 *
 *  Created on: 15.12.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef CAMERAESTIMATION_HPP_
#define CAMERAESTIMATION_HPP_

#include "imageio/ModelLandmark.hpp"

#include "shapemodels/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include <utility>
#include <vector>

namespace shapemodels {

/**
 * A wrapper for the OpenCV solvePnP algorithm, that uses given
 * 2D-3D point correspondences to estimate the extrinsic camera
 * parameters (a rotation and translation).
 * The wrapper takes landmarks in 2D coordinates and finds the
 * corresponding points in a 3DMM using the metadata in the model.
 */
class CameraEstimation  {
public:

	/**
	 * Constructs a new instance of the CameraEstimation algorithm.
	 *
	 * @param[in] MorphableModel The Morphable Model whose shape-model
	 *                           is used to estimate the camera pose.
	 */
	CameraEstimation(/* const? shared_ptr? */MorphableModel morphableModel);

	/**
	 * Takes 2D landmarks, finds the corresponding landmarks in the
	 * 3D model and estimates camera rotation and translation.
	 * Optionally, a vector of vertex-ids can be given for landmarks
	 * not defined in the Morphable Model (e.g. face-contour vertices).
	 *
	 * @param[in] imagePoints Bla
	 * @param[in] intrinsicCameraMatrix Has to be 64F I think!
	 * @param[in] vertexIds Bla
	 * @return Bla R, t
	 */
	std::pair<cv::Mat, cv::Mat> estimate(std::vector<imageio::ModelLandmark> imagePoints, cv::Mat intrinsicCameraMatrix, std::vector<int> vertexIds = std::vector<int>());

private:
	MorphableModel morphableModel;
};

} /* namespace shapemodels */
#endif /* CAMERAESTIMATION_HPP_ */
