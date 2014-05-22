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

#include "opencv2/core/core.hpp"

#include <vector>


namespace morphablemodel {

/**
 * DESC
 *
 * @param[in] morphablemodel Bla
 * @param[in] affineCameraMatrix Has to be 64F I think!
 * @param[in] landmarks Bla
 * @param[in] numCoefficientsToFit Bla
 * @param[in] detectorVariance 2D vari. Should be a vector with one value for every landmark?
 * @param[in] modelVariance 3D vari. Should be a vector with one value for every landmark point in the model?
 * @return Coefficients (alphas) bla
 */
std::vector<float> fitShapeToLandmarksLinear(MorphableModel morphablemodel, cv::Mat affineCameraMatrix, std::vector<int> landmarks, int numCoefficientsToFit=0, float detectorVariance=0.0f, float modelVariance=0.0f);
// variance or sdev?

} /* namespace morphablemodel */
#endif /* LINEARSHAPEFITTING_HPP_ */
