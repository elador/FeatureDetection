/*
 * AffineCameraEstimation.hpp
 *
 *  Created on: 31.12.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef AFFINECAMERAESTIMATION_HPP_
#define AFFINECAMERAESTIMATION_HPP_

#include "imageio/ModelLandmark.hpp"

#include "shapemodels/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include <utility>
#include <vector>

namespace shapemodels {

/**
 * The Gold Standard Algorithm for estimating an affine
 * camera matrix P_A from world to image correspondences.
 * See Algorithm 7.2 in Multiple View Geometry, Hartley &
 * Zisserman, 2nd Edition, 2003
 * 
 */
class AffineCameraEstimation  {
public:

	/**
	 * Constructs a new instance of the CameraEstimation algorithm.
	 *
	 * @param[in] MorphableModel The Morphable Model whose shape-model
	 *                           is used to estimate the camera pose.
	 */
	AffineCameraEstimation(/* const? shared_ptr? */MorphableModel morphableModel);

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
	cv::Mat estimate(std::vector<imageio::ModelLandmark> imagePoints, std::vector<int> vertexIds = std::vector<int>());

	// in: 3x4. Out: 4x4 (z-dir from cross-product)
	static cv::Mat calculateFullMatrix(cv::Mat affineCameraMatrix);

private:
	MorphableModel morphableModel;
};

} /* namespace shapemodels */
#endif /* AFFINECAMERAESTIMATION_HPP_ */
