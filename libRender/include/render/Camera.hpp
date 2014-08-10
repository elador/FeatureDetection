/*
 * Camera.hpp
 *
 *  Created on: 08.08.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include "render/MatrixUtils.hpp"
#include "opencv2/core/core.hpp"

using cv::Vec3f;

namespace render {

class Frustum
{
public:
	Frustum();
	Frustum(float l, float r, float b, float t, float n, float f);

	float l = -1.0f;
	float r = 1.0f;
	float b = -1.0f;
	float t = 1.0f;
	float n = 0.1f;
	float f = 100.0f;
};

/**
 * Desc
 */
class Camera
{
public:
	enum class ProjectionType
	{
		Orthogonal,
		Perspective
	};

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

	cv::Mat getViewMatrix() { // getViewTransform() or getWorldToView(), getWorldToEye(), getWorldToEyeSpaceTransform(), ...
		cv::Mat translate = matrixutils::createTranslationMatrix(-eye[0], -eye[1], -eye[2]);

		cv::Mat rotate = (cv::Mat_<float>(4, 4) <<
			rightVector[0], rightVector[1], rightVector[2], 0.0f,
			upVector[0], upVector[1], upVector[2], 0.0f,
			forwardVector[0], forwardVector[1], forwardVector[2], 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);
		cv::Mat viewTransform = rotate * translate;
		return viewTransform;
	};

	cv::Mat getProjectionMatrix() { // naming, see above
		switch (projectionType)
		{
		case ProjectionType::Orthogonal:
			return matrixutils::createOrthogonalProjectionMatrix(frustum.l, frustum.r, frustum.b, frustum.t, frustum.n, frustum.f);
		case ProjectionType::Perspective:
			return matrixutils::createPerspectiveProjectionMatrix(frustum.l, frustum.r, frustum.b, frustum.t, frustum.n, frustum.f);
		default: // Todo: default necessary? break? what happens? return a Mat()?
			throw std::runtime_error("projectionType is neither Orthogonal nor Perspective. This should never happen.");
		}
	};

	// amount in relation to the basis vector (with unit length)
	// negative amount moves backwards
	void moveForward(float amount) {
		eye += amount * -forwardVector; // we want to go into the gaze direction, and g = -fw
	};
	void moveRight(float amount) {
		eye += amount * rightVector;
	};
	void moveUp(float amount) {
		eye += amount * upVector;
	};

	// makes the camera rotate/look more upwards (i.e. normally the mouse)
	void rotateUp(float amount) {
		gaze += amount * upVector;
	};
	void rotateRight(float amount) {
		gaze += amount * rightVector;
	};

	Frustum& getFrustum() {
		return frustum;
	};

	// input: eye and gaze. Calculate the camera! (completely ignore the angles)
	void updateFixed(const Vec3f& eye, const Vec3f& at, const Vec3f& up = Vec3f(0.0f, 1.0f, 0.0f));
	
	// given the two angles, find the forward, right and up vec. Then, set the eye and gaze in this direction.
	// given hor/verAngle (and eye), calculate the new FwdVec. Then, new right and up. Then also set at-Vec.
	void updateFree(const Vec3f& eye, float horizontalAngle, float verticalAngle, const Vec3f& up = Vec3f(0.0f, 1.0f, 0.0f));
	
	// give a vector with absolute coords where to look at. (given angles + this point, calculate the cam pos.)
	// NOT YET IMPLEMENTED
	void updateFocused(const Vec3f& at, const Vec3f& up = Vec3f(0.0f, 1.0f, 0.0f));
	
	ProjectionType projectionType = ProjectionType::Orthogonal;

	// Todo: move those all back to private, but they should have getters and setters? no?
	Vec3f eye;	///< The eye coordinates (i.e. the position) of the camera, specified by the user. The 'e' point in [Shirley2009, page 146].
	Vec3f gaze;	///< The gaze direction vector, i.e. the direction in which the camera is looking, can be specified by the user. 'g' in [Shirley2009].
	Vec3f up;	///< The upwards direction of the camera, can be specified by the user, usually (0, 1, 0). 't' in [Shirley2009].

private:
	Frustum frustum;		///< Todo doc

	// the camera basis vectors. They are private because they are calculated from the user input (eye-pos e, gaze-dir g, view-up t)
	Vec3f rightVector;		///< The 'u' coordinate axis of the uvw-basis of the camera. See [Shirley2009, page 146] for more details.
	Vec3f upVector;			///< The 'v' coordinate axis.
	Vec3f forwardVector;	///< The 'w' coordinate axis, the opposite to the gaze direction.

};

} /* namespace render */

#endif /* CAMERA_HPP_ */
