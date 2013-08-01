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

using cv::Vec3f;

namespace render {

class Frustum
{
public:
	Frustum();
	Frustum(float l, float r, float b, float t, float n, float f);
	~Frustum() {};

	float l;
	float r;
	float b;
	float t;
	float n;
	float f;
};

/**
 * Desc
 */
class Camera
{
public:
	/**
	 * Constructs a new Camera at the origin looking into -z direction. The Camera
	 * has a default viewing frustum of l=-1, r=1, b=-1, t=1, n=0.1 and f=100.
	 *
	 * @param[in] type The OpenCV type (depth) of the filtered images. If negative, the type remains unchanged.
	 * @param[in] alpha The optional scaling factor.
	 * @param[in] beta The optional delta that is added to the scaled values.
	 */
	Camera();

	/**
	 * Constructs a new Camera at the origin looking into -z direction, with a
	 * given viewing frustum.
	 *
	 * @param[in] Frustum The viewing frustum of the camera.
	 */
	Camera(Frustum frustum);

	/**
	 * Constructs a new Camera at a given position, looking into the direction given
	 * by the horizontal and vertical angle. The viewing frustum is also given.
	 *
	 * @param[in] eyePosition The camera position.
	 * @param[in] horizontalAngle The horizontal angle of the camera viewing direction. (TODO: Further specify this, which direction is plus etc.)
	 * @param[in] verticalAngle The vertical angle of the camera viewing direction.
	 * @param[in] Frustum The viewing frustum of the camera.
	 */
	Camera(Vec3f eyePosition, float horizontalAngle, float verticalAngle, Frustum frustum);

	/**
	 * Constructs a new Camera at a given position, looking into the direction given
	 * by the gazeDirection vector. The viewing frustum is also given.
	 *
	 * @param[in] eyePosition The camera position.
	 * @param[in] gazeDirection The viewing (gaze) direction of the camera.
	 * @param[in] Frustum The viewing frustum of the camera.
	 */
	Camera(Vec3f eyePosition, Vec3f gazeDirection, Frustum frustum);
	
	~Camera();

	float horizontalAngle, verticalAngle;
	//float distanceFromEyeToAt;

	const Vec3f& getEye() const { return eye; }
	const Vec3f& getAt() const { return gaze; }
	const Vec3f& getUp() const { return up; }

	const Vec3f& getForwardVector() const { return forwardVector; }
	const Vec3f& getRightVector() const { return rightVector; }
	const Vec3f& getUpVector() const { return upVector; }

	// input: eye and gaze. Calculate the camera! (completely ignore the angles)
	void updateFixed(const Vec3f& eye, const Vec3f& at, const Vec3f& up = Vec3f(0.0f, 1.0f, 0.0f));
	
	// given the two angles, find the forward, right and up vec. Then, set the eye and gaze in this direction.
	// given hor/verAngle (and eye), calculate the new FwdVec. Then, new right and up. Then also set at-Vec.
	void updateFree(const Vec3f& eye, const Vec3f& up = Vec3f(0.0f, 1.0f, 0.0f));
	
	// give a vector with absolute coords where to look at. (given angles + this point, calculate the cam pos.)
	// NOT YET IMPLEMENTED
	void updateFocused(const Vec3f& at, const Vec3f& up = Vec3f(0.0f, 1.0f, 0.0f));

	// Todo: move those all back to private
	Vec3f eye;	///< The eye coordinates (i.e. the position) of the camera, specified by the user. The 'e' point in [Shirley2009, page 146].
	Vec3f gaze;	///< The gaze direction vector, i.e. the direction in which the camera is looking, can be specified by the user. 'g' in [Shirley2009].
	Vec3f up;	///< The upwards direction of the camera, can be specified by the user, usually (0, 1, 0). 't' in [Shirley2009].

	Frustum frustum;	// TODO make private

private:
	Vec3f rightVector;		///< The 'u' coordinate axis of the uvw-basis of the camera. See [Shirley2009, page 146] for more details.
	Vec3f upVector;			///< The 'v' coordinate axis.
	Vec3f forwardVector;	///< The 'w' coordinate axis, the opposite to the gaze direction.
	
};

} /* namespace render */

#endif /* CAMERA_HPP_ */
