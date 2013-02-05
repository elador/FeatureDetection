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

	std::vector<render::Vertex> vertex;
	//std::vector<cv::Vec3f> vertex;	// g: cv::Vec3f
	//std::vector<cv::Vec3f> normal;	// g: cv::Vec3f
	//std::vector<cv::Vec3f> texcrd;	// g: cv::Vec3f, texture coordinates uvw
	//std::vector<cv::Vec4f> color;	// g: cv::Vec4f, RGBA
	//material

	std::vector< std::array<int, 3> > tvi;	// std::tuple<int, int, int> doesn't work. Use std::array<int, 3> or cv::Vec3i.
	//std::vector< std::array<int, 3> > tni;
	//std::vector< std::array<int, 3> > tti;	// triangle texcrd indices
	std::vector< std::array<int, 3> > tci;
	//tmi
	
	//bool hasVertexColor;
	bool hasTexture;
	std::string textureName;

	render::Texture texture; // optimally, we'd use a TextureManager, or maybe a smart pointer, to not load/store a texture twice if two models use the same texture.

};

}
