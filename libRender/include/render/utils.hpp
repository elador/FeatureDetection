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

	} /* namespace utils */
} /* namespace render */

#endif /* RENDER_UTILS_HPP_ */
