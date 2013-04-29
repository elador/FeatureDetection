/*
 * Landmark.cpp
 *
 *  Created on: 22.03.2013
 *      Author: Patrik Huber
 */

#include "imageio/Landmark.hpp"

namespace imageio {

Landmark::Landmark(string name, Vec3f position, Size2f size, bool visibility) : name(name), position(position), size(size), visibility(visibility)
{
}

Landmark::~Landmark() {}

bool Landmark::isVisible() const
{
	return visibility;
}

string Landmark::getName() const
{
	return name;
}

Vec3f Landmark::getPosition() const
{
	return position;
}

Size2f Landmark::getSize() const
{
	return size;
}

} /* namespace imageio */
