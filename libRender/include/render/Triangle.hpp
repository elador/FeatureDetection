/*!
 * \file Triangle.h
 *
 * \author Patrik Huber
 * \date December 4, 2012
 *
 * [comment here]
 */
#pragma once

#include <array>

#include "render/Vertex.hpp"

namespace render {

class Triangle
{
public:
	Triangle(void);
	Triangle(Vertex v0, Vertex v1, Vertex v2);
	~Triangle(void);

	std::array<Vertex, 3> vertices;
};

}
