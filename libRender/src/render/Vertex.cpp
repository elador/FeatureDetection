/*
 * Vertex.cpp
 *
 *  Created on: 04.12.2012
 *      Author: Patrik Huber
 */

#include "render/Vertex.hpp"

namespace render {

Vertex::Vertex()
{
}

Vertex::Vertex(const cv::Vec4f& position, const cv::Vec3f& color, const cv::Vec2f& texCoord)
{
	this->position = position;
	this->color = color;
	this->texcrd = texCoord;
}

} /* namespace render */
