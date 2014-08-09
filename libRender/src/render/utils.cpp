/*
 * utils.cpp
 *
 *  Created on: 08.06.2014
 *      Author: Patrik Huber
 */
#include "render/utils.hpp"

#include "render/Vertex.hpp"

#include <utility>

using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using cv::Mat;
using std::min;
using std::max;

namespace render {
	namespace utils {

cv::Vec2f clipToScreenSpace(cv::Vec2f clipCoordinates, int screenWidth, int screenHeight)
{
	// Window transform:
	float x_ss = (clipCoordinates[0] + 1.0f) * (screenWidth / 2.0f);
	float y_ss = screenHeight - (clipCoordinates[1] + 1.0f) * (screenHeight / 2.0f); // also flip y; // Qt: Origin top-left. OpenGL: bottom-left.
	return Vec2f(x_ss, y_ss);
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
	clipSpaceV = clipSpaceV / clipSpaceV[3];

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
	
	int minX = max(min(floor(v0.position[0]), min(floor(v1.position[0]), floor(v2.position[0]))), 0.0f);
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

	} /* namespace utils */
} /* namespace render */
