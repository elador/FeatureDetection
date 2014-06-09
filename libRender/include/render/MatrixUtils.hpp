/*
 * MatrixUtils.hpp
 *
 *  Created on: 06.12.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef MATRIXUTILS_HPP_
#define MATRIXUTILS_HPP_

#include "opencv2/core/core.hpp"

// Todo: Class with static methods? Or just functions? I don't know which method is better.

namespace render {

	namespace utils {

		class MatrixUtils
		{
		public:
			static cv::Mat createRotationMatrixX(float angle);
			static cv::Mat createRotationMatrixY(float angle);
			static cv::Mat createRotationMatrixZ(float angle);
			static cv::Mat createScalingMatrix(float sx, float sy, float sz);
			static cv::Mat createTranslationMatrix(float tx, float ty, float tz);
			static cv::Mat createOrthogonalProjectionMatrix(float l, float r, float b, float t, float n, float f);
			static cv::Mat createPerspectiveProjectionMatrix(float l, float r, float b, float t, float n, float f);
			static cv::Mat createPerspectiveProjectionMatrix(float verticalAngle, float aspectRatio, float nearPlane, float farPlane);

		};

		unsigned int getMaxPossibleMipmapsNum(unsigned int width, unsigned int height);	// TODO: This belongs more in a ImageUtils, TextureUtils, or whatever... => render::utils::texturing

	} /* namespace utils */

} /* namespace render */

#endif /* MATRIXUTILS_HPP_ */
