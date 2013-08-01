/*!
 * \file Camera.hpp
 *
 * \author Patrik Huber
 * \date December 5, 2012
 *
 * [comment here]
 */
#pragma once

#include "opencv2/core/core.hpp"

namespace render {

class Camera
{
public:
	Camera(void);
	~Camera(void);

	void init();

	float horizontalAngle, verticalAngle;
	float distanceFromEyeToAt;

	const cv::Vec3f& getEye() const { return eye; }
	const cv::Vec3f& getAt() const { return at; }
	const cv::Vec3f& getUp() const { return up; }

	const cv::Vec3f& getForwardVector() const { return forwardVector; }
	const cv::Vec3f& getRightVector() const { return rightVector; }
	const cv::Vec3f& getUpVector() const { return upVector; }

	void update(int deltaTime);

	void updateFixed(const cv::Vec3f& eye, const cv::Vec3f& at, const cv::Vec3f& up = cv::Vec3f(0.0f, 1.0f, 0.0f));
	void updateFree(const cv::Vec3f& eye, const cv::Vec3f& up = cv::Vec3f(0.0f, 1.0f, 0.0f));
	void updateFocused(const cv::Vec3f& at, const cv::Vec3f& up = cv::Vec3f(0.0f, 1.0f, 0.0f));

	void setFrustum(float l, float r, float t, float b, float n, float f);

private:
	cv::Vec3f eye, at, up;	// (eye is where my eyes are (where I am = position of camera!), at is my target (where I'm looking at), and up is up direction
	cv::Vec3f forwardVector, rightVector, upVector;

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

}
