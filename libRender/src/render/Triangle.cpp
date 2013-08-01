/*
 * Triangle.cpp
 *
 *  Created on: 04.12.2012
 *      Author: Patrik Huber
 */

#include "render/Triangle.hpp"

namespace render {

Triangle::Triangle(void)
{
}

Triangle::Triangle(Vertex v0, Vertex v1, Vertex v2)
{
	vertex[0] = v0;
	vertex[1] = v1;
	vertex[2] = v2;
}

Triangle::~Triangle(void)
{
}

} /* namespace render */
