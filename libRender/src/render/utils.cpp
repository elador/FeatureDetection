/*
 * utils.cpp
 *
 *  Created on: 08.06.2014
 *      Author: Patrik Huber
 */
#include "render/utils.hpp"

#include "render/Vertex.hpp"

#include <algorithm> // min/max
//#include <cmath> // for ceil/floor, should be needed!

using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using cv::Mat;
using std::min;
using std::max;
using std::floor;
using std::ceil;

namespace render {
	namespace utils {

cv::Vec2f clipToScreenSpace(cv::Vec2f clipCoordinates, int screenWidth, int screenHeight)
{
	// Window transform:
	float x_ss = (clipCoordinates[0] + 1.0f) * (screenWidth / 2.0f);
	float y_ss = screenHeight - (clipCoordinates[1] + 1.0f) * (screenHeight / 2.0f); // also flip y; // Qt: Origin top-left. OpenGL: bottom-left.
	return Vec2f(x_ss, y_ss);
	/* Note: What we do here is equivalent to
	   x_w = (x *  vW/2) + vW/2;
	   However, Shirley says we should do:
	   x_w = (x *  vW/2) + (vW-1)/2;
	   (analogous for y)
	   TODO: Check the math!
	*/
}

cv::Vec2f screenToClipSpace(cv::Vec2f screenCoordinates, int screenWidth, int screenHeight)
{
	float x_cs = screenCoordinates[0] / (screenWidth / 2.0f) - 1.0f;
	float y_cs = screenCoordinates[1] / (screenHeight / 2.0f) - 1.0f;
	y_cs *= -1.0f;
	return Vec2f(x_cs, y_cs);
}

cv::Vec3f projectVertex(cv::Vec4f vertex, cv::Mat modelViewProjection, int screenWidth, int screenHeight)
{
	Mat clipSpace = modelViewProjection * Mat(vertex);
	Vec4f clipSpaceV(clipSpace);
	// divide by w:
	clipSpaceV = clipSpaceV / clipSpaceV[3]; // we're in NDC now

	// project from 4D to 2D window position with depth value in z coordinate
	// Viewport transform:
	Vec2f screenSpace = clipToScreenSpace(Vec2f(clipSpaceV[0], clipSpaceV[1]), screenWidth, screenHeight);

	return Vec3f(screenSpace[0], screenSpace[1], clipSpaceV[2]);
}


cv::Rect calculateBoundingBox(Vertex v0, Vertex v1, Vertex v2, int viewportWidth, int viewportHeight)
{
	/* Old, producing artifacts:
	t.minX = max(min(t.v0.position[0], min(t.v1.position[0], t.v2.position[0])), 0.0f);
	t.maxX = min(max(t.v0.position[0], max(t.v1.position[0], t.v2.position[0])), (float)(viewportWidth - 1));
	t.minY = max(min(t.v0.position[1], min(t.v1.position[1], t.v2.position[1])), 0.0f);
	t.maxY = min(max(t.v0.position[1], max(t.v1.position[1], t.v2.position[1])), (float)(viewportHeight - 1));*/
	
	int minX = max(min(floor(v0.position[0]), min(floor(v1.position[0]), floor(v2.position[0]))), 0.0f); // Readded this comment after merge: What about rounding, or rather the conversion from double to int?
	int maxX = min(max(ceil(v0.position[0]), max(ceil(v1.position[0]), ceil(v2.position[0]))), static_cast<float>(viewportWidth - 1));
	int minY = max(min(floor(v0.position[1]), min(floor(v1.position[1]), floor(v2.position[1]))), 0.0f);
	int maxY = min(max(ceil(v0.position[1]), max(ceil(v1.position[1]), ceil(v2.position[1]))), static_cast<float>(viewportHeight - 1));
	return cv::Rect(minX, minY, maxX - minX, maxY - minY);
}

double implicitLine(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2)
{
	return ((double)v1[1] - (double)v2[1])*(double)x + ((double)v2[0] - (double)v1[0])*(double)y + (double)v1[0] * (double)v2[1] - (double)v2[0] * (double)v1[1];
}

bool areVerticesCCWInScreenSpace(const Vertex& v0, const Vertex& v1, const Vertex& v2)
{
	float dx01 = v1.position[0] - v0.position[0];
	float dy01 = v1.position[1] - v0.position[1];
	float dx02 = v2.position[0] - v0.position[0];
	float dy02 = v2.position[1] - v0.position[1];

	return (dx01*dy02 - dy01*dx02 < 0.0f); // Original: (dx01*dy02 - dy01*dx02 > 0.0f). But: OpenCV has origin top-left, y goes down
}

float radiansToDegrees(float radians)
{
	return radians * static_cast<float>(180 / CV_PI);
}

float degreesToRadians(float degrees)
{
	return degrees * static_cast<float>(CV_PI / 180);
}

float cot(float x)
{
	return std::tan((CV_PI / 2.0f) - x);
}

float fovyToFocalLength(float fovy, float height)
{
	// The actual equation is: $ cot(fov/2) = f/(h/2) $
	// equivalent to: $ f = (h/2) * cot(fov/2) $
	// Now I assume that in OpenGL, h = 2 (-1 to 1), so this simplifies to
	// $ f = cot(fov/2) $, which corresponds with http://wiki.delphigl.com/index.php/gluPerspective.
	// It also coincides with Rafaels Formula.

	return (cot(degreesToRadians(fovy) / 2) * (height / 2.0f));
}

float focalLengthToFovy(float focalLength, float height)
{
	return radiansToDegrees(2.0f * std::atan2(height, 2.0f * focalLength)); // both are always positive, so atan() should do it as well?
}

cv::Vec3f eulerAnglesFromRotationMatrix(cv::Mat R)
{
	cv::Vec3f eulerAngles;
	eulerAngles[0] = std::atan2(R.at<float>(2, 1), R.at<float>(2, 2)); // r_32, r_33. Theta_x.
	eulerAngles[1] = std::atan2(-R.at<float>(2, 0), std::sqrt(std::pow(R.at<float>(2, 1), 2) + std::pow(R.at<float>(2, 2), 2))); // r_31, sqrt(r_32^2 + r_33^2). Theta_y.
	eulerAngles[2] = std::atan2(R.at<float>(1, 0), R.at<float>(0, 0)); // r_21, r_11. Theta_z.
	return eulerAngles;
}

	} /* namespace utils */
} /* namespace render */
