/*
 * LinearShapeFitting.hpp
 *
 *  Created on: 22.05.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LINEARSHAPEFITTING_HPP_
#define LINEARSHAPEFITTING_HPP_

#include "morphablemodel/MorphableModel.hpp"

#include "imageio/ModelLandmark.hpp"

#include "opencv2/core/core.hpp"

//#include "boost/optional.hpp" // Weird: Why does it compile without this header? (Win64)

#include <vector>

namespace fitting {

/**
 * Fits the shape of a Morphable Model to .. (i.e. estimates the ML sol of the coeffs...) as in [1].
 * linear, closed-form solution fitting of the shape, with regul. (prior to mean)
 * The fitting is done in clip-coords, the given cam matrix should transform there.
 *
 * [1] O. Aldrian & W. Smith, Inverse Rendering of Faces with a 3D Morphable Model, PAMI 2013.
 *
 * @param[in] morphableModel The Morphable Model whose shape (coefficients) are fitted.
 * @param[in] affineCameraMatrix A 3x4 affine camera matrix from world to clip-space (should probably be of type CV_32FC1 as all our calculations are done with float).
 * @param[in] landmarks 2D landmarks from an image, given in clip-coordinates.
 * @param[in] lambda The regularisation parameter (weight of the prior towards the mean).
 * @param[in] numCoefficientsToFit How many shape-coefficients to fit (all others will stay 0). Not used yet.
 * @param[in] detectorStandardDeviation The 2D standard deviation of the landmark detector used. Should be a vector with one value for every landmark? TODO: Add if we should give this in pixels, % of IED, and in img or clip-space.
 * @param[in] modelStandardDeviation The 3D standard deviation of each corresponding point (vertex) in the 3D model. Should be a vector with one value for every landmark point in the model? TODO: Also mention what unit.
 * @return The fitted shape-coefficients (alphas).
 */
std::vector<float> fitShapeToLandmarksLinear(morphablemodel::MorphableModel morphableModel, cv::Mat affineCameraMatrix, std::vector<imageio::ModelLandmark> landmarks, float lambda=20.0f, boost::optional<int> numCoefficientsToFit=boost::optional<int>(), boost::optional<float> detectorStandardDeviation=boost::optional<float>(), boost::optional<float> modelStandardDeviation=boost::optional<float>());

} /* namespace fitting */
#endif /* LINEARSHAPEFITTING_HPP_ */
