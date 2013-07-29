/*
 * Camera.cpp
 *
 *  Created on: 05.12.2012
 *      Author: Patrik Huber
 */

#include "render/Camera.hpp"
#include "render/MatrixUtils.hpp"

using cv::Mat;
using cv::Vec4f;

namespace render {
	
Frustum::Frustum()
{
	l = -1.0f;
	r = 1.0f;
	b = -1.0f;
	t = 1.0f;
	n = 0.1f;
	f = 100.0f;
}

Frustum::Frustum(float l, float r, float b, float t, float n, float f)
{
	this->l = l;
	this->r = r;
	this->b = b;
	this->t = t;
	this->n = n;
	this->f = f;
}

Camera::Camera()
{
	horizontalAngle = 0.0f;
	verticalAngle = 0.0f;
	updateFixed(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f));
}

Camera::Camera(Frustum frustum)
{
	this->frustum = frustum;
	horizontalAngle = 0.0f;
	verticalAngle = 0.0f;
	updateFixed(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f));
}

Camera::Camera(Vec3f eyePosition, float horizontalAngle, float verticalAngle, Frustum frustum)
{
	this->frustum = frustum;
	this->horizontalAngle = horizontalAngle;
	this->verticalAngle = verticalAngle;
	updateFree(eyePosition);
}

Camera::Camera(Vec3f eyePosition, Vec3f gazeDirection, Frustum frustum)
{
	this->frustum = frustum;
	horizontalAngle = 0.0f;
	verticalAngle = 0.0f;
	updateFixed(eyePosition, gazeDirection);
}

Camera::~Camera()
{
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

void Camera::updateFree(const Vec3f& eye, const Vec3f& up)
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
	throw("Sorry, not yet implemented!");
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
