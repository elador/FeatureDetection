/*
 * Camera.cpp
 *
 *  Created on: 08.08.2014
 *      Author: Patrik Huber
 */

#include "render/Camera.hpp"

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

void Camera::updateFixed(const Vec3f& eye, const Vec3f& gaze, const Vec3f& up /*= Vec3f(0.0f, 1.0f, 0.0f)*/)
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

// Hmm, I think kind of in both of these functions it can happen that some vectors are not orthogonal? That up vector is a bit strange?
void Camera::updateFree(const Vec3f& eye, float horizontalAngle, float verticalAngle, const Vec3f& up /*= Vec3f(0.0f, 1.0f, 0.0f)*/)
{
	// Set new basis axes, i.e. first a new forwardVector and rightVector, rotated by the given horizontal and vertical angles:
	Mat transformMatrix = matrixutils::createRotationMatrixY(horizontalAngle) * matrixutils::createRotationMatrixX(verticalAngle);
	Mat rotatedForwardVector = transformMatrix * Mat(Vec4f(0.0f, 0.0f, 1.0f, 1.0f));
	forwardVector[0] = rotatedForwardVector.at<float>(0, 0);	// This rotates the standard forward-vector (0, 0, 1) with the rotation
	forwardVector[1] = rotatedForwardVector.at<float>(1, 0);	// matrix and sets the new forward-vector accordingly
	forwardVector[2] = rotatedForwardVector.at<float>(2, 0);

	Mat rotatedRightVector = transformMatrix * Mat(Vec4f(1.0f, 0.0f, 0.0f, 1.0f));
	rightVector[0] = rotatedRightVector.at<float>(0, 0);	// Same for the right-vector (1, 0, 0)
	rightVector[1] = rotatedRightVector.at<float>(1, 0);
	rightVector[2] = rotatedRightVector.at<float>(2, 0);

	upVector = forwardVector.cross(rightVector);

	// New basis is set. Now let's do ....?
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
