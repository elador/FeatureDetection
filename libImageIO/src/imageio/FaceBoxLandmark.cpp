/*
 * FaceBoxLandmark.cpp
 *
 *  Created on: 23.03.2013
 *      Author: Patrik Huber
 */

#include "imageio/FaceBoxLandmark.hpp"

namespace imageio {

FaceBoxLandmark::FaceBoxLandmark(string name) : Landmark(name)
{
}

FaceBoxLandmark::FaceBoxLandmark(string name, Vec3f position, float width, float height, bool visibility) : Landmark(name, position, visibility), width(width), height(height)
{
}

FaceBoxLandmark::~FaceBoxLandmark() {}

float FaceBoxLandmark::getWidth() const
{
	return width;
}

float FaceBoxLandmark::getHeight() const
{
	return height;
}


} /* namespace imageio */
