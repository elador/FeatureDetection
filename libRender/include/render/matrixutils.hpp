/*
 * matrixutils.hpp
 *
 *  Created on: 06.12.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef MATRIXUTILS_HPP_
#define MATRIXUTILS_HPP_

#include "opencv2/core/core.hpp"

namespace render {

namespace matrixutils {

cv::Mat createRotationMatrixX(float angle);
cv::Mat createRotationMatrixY(float angle);
cv::Mat createRotationMatrixZ(float angle);
cv::Mat createScalingMatrix(float sx, float sy, float sz);
cv::Mat createTranslationMatrix(float tx, float ty, float tz);
cv::Mat createOrthogonalProjectionMatrix(float l, float r, float b, float t, float n, float f);
cv::Mat createPerspectiveProjectionMatrix(float l, float r, float b, float t, float n, float f);
cv::Mat createPerspectiveProjectionMatrix(float verticalAngle, float aspectRatio, float nearPlane, float farPlane);

unsigned int getMaxPossibleMipmapsNum(unsigned int width, unsigned int height);	// TODO: This belongs more in a ImageUtils, TextureUtils, or whatever... => render::utils::texturing

} /* namespace matrixutils */

} /* namespace render */

#endif /* MATRIXUTILS_HPP_ */
