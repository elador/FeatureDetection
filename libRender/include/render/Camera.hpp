/*
 * Camera.hpp
 *
 *  Created on: 05.12.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include "opencv2/core/core.hpp"

namespace render {

/**
 * Desc
 */
class Camera
{
public:
	// Todo: Create 2 c'tors.
	// - takes 2 angl + eye + frustum, does updateFree
	// - takes eye + gaze + frustum, does updateFixed.
	// (- later a 3rd for focused)
	Camera(void);
	~Camera(void);

	// sets up cam, angles = 0, eye 0 0 0 and gaze along -z.
	void init();

	float horizontalAngle, verticalAngle;
	//float distanceFromEyeToAt;

	const cv::Vec3f& getEye() const { return eye; }
	const cv::Vec3f& getAt() const { return gaze; }
	const cv::Vec3f& getUp() const { return up; }

	const cv::Vec3f& getForwardVector() const { return forwardVector; }
	const cv::Vec3f& getRightVector() const { return rightVector; }
	const cv::Vec3f& getUpVector() const { return upVector; }

	void update(int deltaTime);

	void updateFixed(const cv::Vec3f& eye, const cv::Vec3f& at, const cv::Vec3f& up = cv::Vec3f(0.0f, 1.0f, 0.0f));
	void updateFree(const cv::Vec3f& eye, const cv::Vec3f& up = cv::Vec3f(0.0f, 1.0f, 0.0f));
	void updateFocused(const cv::Vec3f& at, const cv::Vec3f& up = cv::Vec3f(0.0f, 1.0f, 0.0f));

	void setFrustum(float l, float r, float t, float b, float n, float f);

	// Todo: move those all back to private
	// viewer position and orientation. specified by the user.
	cv::Vec3f eye;	// 'e' eye is where my eyes are (where I am = position of camera!)
	cv::Vec3f gaze; // 'g', gaze, lookAt, at is my target (where I'm looking at)
	cv::Vec3f up; // 't', up is up direction

	// uvw-basis, updated per frame
	cv::Vec3f forwardVector; // w (opposite to gaze!)
	cv::Vec3f rightVector; // u
	cv::Vec3f upVector; // v
private:


	struct Frustum {
		float l;
		float r;
		float t;
		float b;
		float n;
		float f;
	};

public:
	Frustum frustum;	// TODO make private
};

} /* namespace render */

#endif /* CAMERA_HPP_ */
