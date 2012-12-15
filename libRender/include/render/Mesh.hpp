/*!
 * \file Mesh.hpp
 *
 * \author Patrik Huber
 * \date December 12, 2012
 *
 * [comment here]
 */
#pragma once

#include "render/Vertex.hpp"
#include "render/Triangle.hpp"
#include "render/Texture.hpp"

#include <vector>
#include <array>
#include <string>

namespace render {

class Mesh
{
public:
	Mesh(void);
	~Mesh(void);

	std::vector<cv::Vec3f> vertex;
	std::vector<cv::Vec3f> normal;
	std::vector<cv::Vec3f> texcrd;	// texture coordinates uvw
	std::vector<cv::Vec4f> color;	// RGBA
	//material

	std::vector< std::array<int, 3> > tvi;	// std::tuple<int, int, int> doesn't work. Use std::array<int, 3> or cv::Vec3i.
	std::vector< std::array<int, 3> > tni;
	std::vector< std::array<int, 3> > tti;	// triangle texcrd indices
	std::vector< std::array<int, 3> > tci;
	//tmi
	
	//bool hasVertexColor;
	bool hasTexture;
	std::string textureName;

	std::vector<render::Vertex> vertices;
	//std::vector<render::Triangle> triangleList; // --> make the renderer work with indices. How does gravis do it?

	render::Texture texture; // optimally, we'd use a TextureManager, or maybe a smart pointer, to not load/store a texture twice if two models use the same texture.

};

}
