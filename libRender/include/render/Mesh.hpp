/*
 * Mesh.hpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef MESH_HPP_
#define MESH_HPP_

#include "render/Vertex.hpp"
#include "render/Triangle.hpp"
#include "render/Texture.hpp"

#include <vector>
#include <array>
#include <string>
#include <memory>

namespace render {

/**
 * Desc
 * TODO: Make stuff private?
 */
class Mesh
{
public:

	std::vector<render::Vertex> vertex;
	// To not have to copy the vertices, for OpenGL, the option below would be better:
	// But what about the whole pipeline (e.g. SW-Rend.), is it beneficial having a Vertex (& Triangle)-class?
	//std::vector<cv::Vec3f> vertex;	// g: cv::Vec3f
	//std::vector<cv::Vec3f> normal;	// g: cv::Vec3f
	//std::vector<cv::Vec3f> texcrd;	// g: cv::Vec3f, texture coordinates uvw
	//std::vector<cv::Vec4f> color;	// g: cv::Vec4f, RGBA
	//material

	std::vector<std::array<int, 3>> tvi;	// std::tuple<int, int, int> doesn't work. Use std::array<int, 3> or cv::Vec3i.
	//std::vector< std::array<int, 3> > tni;
	//std::vector< std::array<int, 3> > tti;	// triangle texcrd indices
	std::vector<std::array<int, 3>> tci;
	//tmi
	
	//bool hasVertexColor;
	bool hasTexture = false;
	std::string textureName;

	std::shared_ptr<render::Texture> texture; // optimally, we'd use a TextureManager, or maybe a smart pointer, to not load/store a texture twice if two models use the same texture.

	// TODO Doc
	// obj with vertex-coloring (not officially supported but works eg in meshlab)
	static void writeObj(Mesh mesh, std::string filename);

};

} /* namespace render */

#endif /* MESH_HPP_ */
