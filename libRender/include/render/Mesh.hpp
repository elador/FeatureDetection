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
#include <tuple>
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

	std::vector< std::tuple<int, int, int> > tvi;
	std::vector< std::tuple<int, int, int> > tni;
	std::vector< std::tuple<int, int, int> > tti;	// triangle texcrd indices
	std::vector< std::tuple<int, int, int> > tci;
	//tmi
	
	//bool hasVertexColor;
	bool hasTexture;
	std::string textureName;

	std::vector<render::Vertex> vertices;
	int triangleIndices;
	std::vector<render::Triangle> triangleList; // --> make the renderer work with indices. How does gravis do it?

	render::Texture texture; // optimally, we'd use a TextureManager, or maybe a smart pointer, to not load/store a texture twice if two models use the same texture.

};

}
