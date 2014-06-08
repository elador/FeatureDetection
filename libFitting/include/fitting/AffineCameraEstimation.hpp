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

#include "morphablemodel/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include <vector>

namespace fitting {

/**
 * The Gold Standard Algorithm for estimating an affine
 * camera matrix from world to image correspondences.
 * See Algorithm 7.2 in Multiple View Geometry, Hartley &
 * Zisserman, 2nd Edition, 2003.
 * Optionally, a vector of vertex-ids can be given for landmarks
 * not defined in the Morphable Model (e.g. face-contour vertices) - not implemented yet!
 *
 * @param[in] imagePoints A list of 2D image points
 * @param[in] morphableModel The 3D model whose correspondences are used to estimate the camera
 * @param[in] vertexIds An optional list of vertex ids if not all given imagePoints have a corresponding point in the model. TODO: Should this better be a map that is used in addition to the standard lookup?
 * @return A 3x4 affine camera matrix (the third row is [0, 0, 0, 1]).
 */
cv::Mat estimateAffineCamera(std::vector<imageio::ModelLandmark> imagePoints, morphablemodel::MorphableModel morphableModel, std::vector<int> vertexIds=std::vector<int>());

/**
 * Takes a 3x4 affine camera matrix, calculates the
 * viewing direction using the cross-product of the first
 * and second row and returns a 4x4 affine camera matrix.
 * Caution: The camera might look the wrong way, we don't
 * check for that.
 *
 * @param[in] affineCameraMatrix A 3x4 affine camera matrix
 * @return A 4x4 affine camera matrix (the fourth row is [0, 0, 0, 1]).
 */
cv::Mat calculateAffineZDirection(cv::Mat affineCameraMatrix);

/**
 * Projects a point from world coordinates to screen coordinates.
 * First, an estimated affine camera matrix is used to transform
 * the point to clip space. Second, the point is transformed to
 * screen coordinates using the window transform. The window transform
 * also flips the y-axis (the image origin is top-left, while in
 * clip space top is +1 and bottom is -1).
 *
 * @param[in] vertex Todo
 * @param[in] affineCameraMatrix 3x4.
 * @param[in] screenWidth Todo
 * @param[in] screenHeight Todo
 * @return A vector with x and y coordinates transformed to clip space.
 */
cv::Vec2f projectAffine(cv::Vec3f vertex, cv::Mat affineCameraMatrix, int screenWidth, int screenHeight); // Could go into libRender, but Render doesn't have a dependency on libImageIO (yet). But we don't need ImageIO.

} /* namespace fitting */
#endif /* AFFINECAMERAESTIMATION_HPP_ */
