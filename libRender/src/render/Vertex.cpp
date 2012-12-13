/*!
 * \file Vertex.cpp
 *
 * \author Patrik Huber
 * \date December 4, 2012
 *
 * [comment here]
 */

#include "render/Vertex.hpp"

namespace render {

Vertex::Vertex(void)
{
	// default constructor should be ok, initializes all cv::Vec with zeros
}

Vertex::Vertex(const cv::Vec4f& position, const cv::Vec3f& color, const cv::Vec2f& texCoord)
{
	this->position = position;
	this->color = color;
	this->texCoord = texCoord;
}

Vertex::~Vertex(void)
{
}

}
