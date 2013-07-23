/*
 * Camera.cpp
 *
 *  Created on: 05.12.2012
 *      Author: Patrik Huber
 */

#include "render2/Camera.hpp"
#include "render2/MatrixUtils.hpp"

namespace render {
	
Camera::Camera(void) : horizontalAngle(0.0f), verticalAngle(0.0f), distanceFromEyeToAt(1.0f)
{
}


Camera::~Camera(void)
{
}

void Camera::init()
{
	//this->verticalAngle = -CV_PI/4.0f;
	this->verticalAngle = 0.0f;
	this->updateFixed(cv::Vec3f(0.0f, 0.0f, 3.0f), cv::Vec3f(0.0f, 0.0f, 0.0f));	// "at" initialization doesn't matter
}

void Camera::update(int deltaTime)	// Hmm this doesn't really belong here, it's application dependent. But ok for now.
{
	float speed = 0.02f;
	cv::Vec3f eye = this->getEye();

	/*eye += speed * deltaTime * this->getForwardVector();	// 'w' key
	eye -= speed * deltaTime * this->getForwardVector();	// 's' key
	eye -= speed * deltaTime * this->getRightVector();	// 'a' key
	eye += speed * deltaTime * this->getRightVector();	// 'd' key
	*/
	this->updateFree(eye);
}

void Camera::updateFixed(const cv::Vec3f& eye, const cv::Vec3f& at, const cv::Vec3f& up)
{
	this->forwardVector = at - eye;
	this->forwardVector /= cv::norm(forwardVector, cv::NORM_L2);

	this->eye = eye;
	this->at = at;
	this->up = up;
}


void Camera::updateFree(const cv::Vec3f& eye, const cv::Vec3f& up)
{
	cv::Mat transformMatrix = render::utils::MatrixUtils::createRotationMatrixY(horizontalAngle) * render::utils::MatrixUtils::createRotationMatrixX(verticalAngle);
	cv::Mat tmpRes = transformMatrix * cv::Mat(cv::Vec4f(0.0f, 0.0f, -1.0f, 1.0f));
	//cv::Mat tmp = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, -1.0f, 0.0f);
	//cv::Mat tmpRes = tmp * transformMatrix;
	forwardVector[0] = tmpRes.at<float>(0, 0);	// This rotates the standard forward-vector (0, 0, -1) with the rotation
	forwardVector[1] = tmpRes.at<float>(1, 0);	// matrix and sets the new forward-vector accordingly (?)
	forwardVector[2] = tmpRes.at<float>(2, 0);

	cv::Mat tmpRes2 = transformMatrix * cv::Mat(cv::Vec4f(1.0f, 0.0f, 0.0f, 1.0f));
	//cv::Mat tmp2 = (cv::Mat_<float>(1, 4) << 1.0f, 0.0f, 0.0f, 0.0f);
	//cv::Mat tmpRes2 = tmp2 * transformMatrix;
	rightVector[0] = tmpRes2.at<float>(0, 0);
	rightVector[1] = tmpRes2.at<float>(1, 0);
	rightVector[2] = tmpRes2.at<float>(2, 0);
	//rightVector = cv::Vec3f(1.0f, 0.0f, 0.0f) * transformMatrix;

	upVector = rightVector.cross(forwardVector);

	this->eye = eye;
	this->at = eye + forwardVector;
	this->up = up;
}



void Camera::updateFocused(const cv::Vec3f& at, const cv::Vec3f& up)
{
	cv::Mat transformMatrix = render::utils::MatrixUtils::createRotationMatrixY(horizontalAngle) * render::utils::MatrixUtils::createRotationMatrixX(verticalAngle);

	cv::Mat tmp = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, -1.0f, 0.0f);
	cv::Mat tmpRes = tmp * transformMatrix;
	forwardVector[0] = tmpRes.at<float>(0, 0);
	forwardVector[1] = tmpRes.at<float>(0, 1);
	forwardVector[2] = tmpRes.at<float>(0, 2);

	cv::Mat tmp2 = (cv::Mat_<float>(1, 4) << 1.0f, 0.0f, 0.0f, 0.0f);
	cv::Mat tmpRes2 = tmp2 * transformMatrix;
	rightVector[0] = tmpRes2.at<float>(0, 0);
	rightVector[1] = tmpRes2.at<float>(0, 1);
	rightVector[2] = tmpRes2.at<float>(0, 2);
	//rightVector = cv::Vec3f(1.0f, 0.0f, 0.0f) * transformMatrix;
	
	upVector = rightVector.cross(forwardVector);

	this->eye = at - forwardVector*distanceFromEyeToAt;
	this->at = at;
	this->up = up;
}

void Camera::setFrustum( float l, float r, float t, float b, float n, float f )
{
	frustum.l = l;
	frustum.r = r;
	frustum.t = t;
	frustum.b = b;
	frustum.n = n;
	frustum.f = f;
}

} /* namespace render */
