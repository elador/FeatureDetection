/*
 * utils.cpp
 *
 *  Created on: 08.06.2014
 *      Author: Patrik Huber
 */
#include "render/utils.hpp"

cv::Vec2f clipToScreenSpace(cv::Vec2f clipCoordinates, int screenWidth, int screenHeight)
{
	// Window transform:
	float x_ss = (clipCoordinates[0] + 1.0f) * (screenWidth / 2.0f);
	float y_ss = screenHeight - (clipCoordinates[1] + 1.0f) * (screenHeight / 2.0f); // also flip y
	return cv::Vec2f(x_ss, y_ss);
}

cv::Vec2f screenToClipSpace(cv::Vec2f screenCoordinates, int screenWidth, int screenHeight)
{
	float x_cs = screenCoordinates[0] / (screenWidth / 2.0f) - 1.0f;
	float y_cs = screenCoordinates[1] / (screenHeight / 2.0f) - 1.0f;
	y_cs *= -1.0f;
	return cv::Vec2f(x_cs, y_cs);
}
