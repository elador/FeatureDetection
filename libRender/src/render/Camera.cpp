/*
 * Camera.cpp
 *
 *  Created on: 08.08.2014
 *      Author: Patrik Huber
 */

#include "render/Camera.hpp"
#include "render/MatrixUtils.hpp"

using cv::Mat;
using cv::Vec4f;

namespace render {
	
Frustum::Frustum()
{
}

Frustum::Frustum(float l, float r, float b, float t, float n, float f) : l(l), r(r), b(b), t(t), n(n), f(f)
{
}

Camera::Camera()
{
	updateFixed(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f));
}

Camera::Camera(Frustum frustum) : frustum(frustum)
{
	updateFixed(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f));
}

Camera::Camera(Vec3f eyePosition, float horizontalAngle, float verticalAngle, Frustum frustum) : frustum(frustum)
{
	updateFree(eyePosition, horizontalAngle, verticalAngle);
}

Camera::Camera(Vec3f eyePosition, Vec3f gazeDirection, Frustum frustum) : frustum(frustum)
{
	updateFixed(eyePosition, gazeDirection);
}

void Camera::updateFixed(const Vec3f& eye, const Vec3f& gaze, const Vec3f& up)
{
	this->eye = eye;
	this->gaze = gaze;
	this->up = up;

	this->forwardVector = -gaze;
	this->forwardVector /= cv::norm(forwardVector, cv::NORM_L2);

	this->rightVector = up.cross(forwardVector);
	this->rightVector /= cv::norm(rightVector, cv::NORM_L2);

	this->upVector = forwardVector.cross(rightVector);
}

void Camera::updateFree(const Vec3f& eye, float horizontalAngle, float verticalAngle, const Vec3f& up /*= Vec3f(0.0f, 1.0f, 0.0f)*/)
{
	Mat transformMatrix = render::utils::MatrixUtils::createRotationMatrixY(horizontalAngle) * render::utils::MatrixUtils::createRotationMatrixX(verticalAngle);
	Mat tmpRes = transformMatrix * Mat(Vec4f(0.0f, 0.0f, 1.0f, 1.0f));
	forwardVector[0] = tmpRes.at<float>(0, 0);	// This rotates the standard forward-vector (0, 0, 1) with the rotation
	forwardVector[1] = tmpRes.at<float>(1, 0);	// matrix and sets the new forward-vector accordingly
	forwardVector[2] = tmpRes.at<float>(2, 0);

	Mat tmpRes2 = transformMatrix * Mat(Vec4f(1.0f, 0.0f, 0.0f, 1.0f));
	rightVector[0] = tmpRes2.at<float>(0, 0);
	rightVector[1] = tmpRes2.at<float>(1, 0);
	rightVector[2] = tmpRes2.at<float>(2, 0);

	this->upVector = forwardVector.cross(rightVector);

	this->eye = eye;
	this->gaze = -forwardVector;
	this->up = up;
}

void Camera::updateFocused(const Vec3f& lookAt, const Vec3f& up)
{
	throw std::runtime_error("Sorry, not yet implemented!");
	/*
	Mat transformMatrix = render::utils::MatrixUtils::createRotationMatrixY(horizontalAngle) * render::utils::MatrixUtils::createRotationMatrixX(verticalAngle);

	Mat tmp = (Mat_<float>(1, 4) << 0.0f, 0.0f, -1.0f, 0.0f);
	Mat tmpRes = tmp * transformMatrix;
	forwardVector[0] = tmpRes.at<float>(0, 0);
	forwardVector[1] = tmpRes.at<float>(0, 1);
	forwardVector[2] = tmpRes.at<float>(0, 2);

	Mat tmp2 = (Mat_<float>(1, 4) << 1.0f, 0.0f, 0.0f, 0.0f);
	Mat tmpRes2 = tmp2 * transformMatrix;
	rightVector[0] = tmpRes2.at<float>(0, 0);
	rightVector[1] = tmpRes2.at<float>(0, 1);
	rightVector[2] = tmpRes2.at<float>(0, 2);
	//rightVector = Vec3f(1.0f, 0.0f, 0.0f) * transformMatrix;
	
	upVector = rightVector.cross(forwardVector);

	this->eye = lookAt - forwardVector*distanceFromEyeToAt;
	this->gaze = lookAt;
	this->up = up;
	*/
}


} /* namespace render */
