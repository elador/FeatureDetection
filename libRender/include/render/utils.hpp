/*
 * utils.hpp
 *
 *  Created on: 08.06.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef RENDER_UTILS_HPP_
#define RENDER_UTILS_HPP_

#include "opencv2/core/core.hpp"

// Forward declarations:
namespace render {
	class Vertex;
}

/**
 * The render::utils namespace contains utility
 * functions for miscellaneous rendering tasks.
 */
namespace render {
	namespace utils {

/**
 * Transforms a point from clip space ([-1, 1] x [-1, 1]) to
 * image (screen) coordinates, i.e. the window transform.
 * Note that the y-coordinate is flipped because the image origin
 * is top-left while in clip space top is +1 and bottom is -1.
 * No z-division is performed.
 *
 * @param[in] clipCoordinates Todo
 * @param[in] screenWidth Todo
 * @param[in] screenHeight Todo
 * @return A vector with x and y coordinates transformed to clip space.
 */
cv::Vec2f clipToScreenSpace(cv::Vec2f clipCoordinates, int screenWidth, int screenHeight);

/**
 * Transforms a point from image (screen) coordinates to
 * clip space ([-1, 1] x [-1, 1]).
 * Note that the y-coordinate is flipped because the image origin
 * is top-left while in clip space top is +1 and bottom is -1.
 *
 * @param[in] screenCoordinates Todo
 * @param[in] screenWidth Todo
 * @param[in] screenHeight Todo
 * @return A vector with x and y coordinates transformed to clip space.
 */
cv::Vec2f screenToClipSpace(cv::Vec2f screenCoordinates, int screenWidth, int screenHeight);

/**
 * Returns the 2D bounding box of the triangle given by the
 * three vertices. Uses the x and y components of the vertices,
 * assuming they are in screen space. 
 * Rounded with floor and ceil respectively to integer values.
 * Also clips against zero and the width and height of the
 * viewport.
 *
 * Note: Not 100% happy about this, we should rather
 * return (minX, maxX, minY, maxY) instead of (x, y, w, h).
 *
 * @param[in] v0 The 0th vertex of a triangle
 * @param[in] v1 The 1st vertex of a triangle
 * @param[in] v2 The 2nd vertex of a triangle
 * @param[in] viewportWidth The TODO
 * @param[in] viewportHeight The TODO
 * @return The bounding box encompassing the triangle given.
 */
cv::Rect calculateBoundingBox(Vertex v0, Vertex v1, Vertex v2, int viewportWidth, int viewportHeight);

/**
 * Todo.
 *
 * @param[in] all Todo
 * @return Todo.
 */
double implicitLine(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2);

/**
 * Todo.
 *
 * Note: For screens with origin top-left, y goes down, like OpenCV
 *
 * @param[in] all Todo
 * @return Todo.
 */
bool areVerticesCCWInScreenSpace(const Vertex& v0, const Vertex& v1, const Vertex& v2);

	} /* namespace utils */
} /* namespace render */

#endif /* RENDER_UTILS_HPP_ */
