/*
 * MatrixUtils.cpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */

#include "render2/MatrixUtils.hpp"

namespace render {
	namespace utils {

cv::Mat MatrixUtils::createRotationMatrixX(float angle)
{
	cv::Mat rotX = (cv::Mat_<float>(4,4) << 
		1.0f,				0.0f,				0.0f,				0.0f,
		0.0f,				std::cos(angle),	-std::sin(angle),	0.0f,
		0.0f,				std::sin(angle),	std::cos(angle),	0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);
	return rotX;
}

cv::Mat MatrixUtils::createRotationMatrixY(float angle)
{
	cv::Mat rotY = (cv::Mat_<float>(4,4) << 
		std::cos(angle),	0.0f,				std::sin(angle),	0.0f,
		0.0f,				1.0f,				0.0f,				0.0f,
		-std::sin(angle),	0.0f,				std::cos(angle),	0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);
	return rotY;
}

cv::Mat MatrixUtils::createRotationMatrixZ(float angle)
{
	cv::Mat rotZ = (cv::Mat_<float>(4,4) << 
		std::cos(angle),	-std::sin(angle),	0.0f,				0.0f,
		std::sin(angle),	std::cos(angle),	0.0f,				0.0f,
		0.0f,				0.0f,				1.0f,				0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);
	return rotZ;
}

cv::Mat MatrixUtils::createScalingMatrix(float sx, float sy, float sz)
{
	cv::Mat scaling = (cv::Mat_<float>(4,4) << 
		sx,					0.0f,				0.0f,				0.0f,
		0.0f,				sy,					0.0f,				0.0f,
		0.0f,				0.0f,				sz,					0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);
	return scaling;
}

cv::Mat MatrixUtils::createTranslationMatrix(float tx, float ty, float tz)
{
	cv::Mat translation = (cv::Mat_<float>(4,4) << 
		1.0f,				0.0f,				0.0f,				tx,
		0.0f,				1.0f,				0.0f,				ty,
		0.0f,				0.0f,				1.0f,				tz,
		0.0f,				0.0f,				0.0f,				1.0f);
	return translation;
}

	} /* namespace utils */
} /* namespace render */
