/*
 * MeshUtils.cpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */

#include "render/MeshUtils.hpp"

#include "opencv2/core/core.hpp"

#include <array>
#include <iostream>
#include <fstream>

namespace render {
	namespace utils {

Mesh MeshUtils::createCube()
{
	Mesh cube;
	cube.vertex.resize(24);

	for (int i = 0; i < 24; i++)
		cube.vertex[i].color = cv::Vec3f(1.0f, 1.0f, 0.0f);

	cube.vertex[0].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[0].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[1].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[1].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[2].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[2].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[3].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[3].texcrd = cv::Vec2f(1.0f, 0.0f);

	cube.vertex[4].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[4].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[5].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[5].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[6].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[6].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[7].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[7].texcrd = cv::Vec2f(1.0f, 0.0f);

	cube.vertex[8].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[8].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[9].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[9].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[10].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[10].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[11].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[11].texcrd = cv::Vec2f(1.0f, 0.0f);

	cube.vertex[12].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[12].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[13].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[13].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[14].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[14].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[15].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[15].texcrd = cv::Vec2f(1.0f, 0.0f);

	cube.vertex[16].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[16].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[17].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[17].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[18].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[18].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[19].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[19].texcrd = cv::Vec2f(1.0f, 0.0f);

	cube.vertex[20].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[20].texcrd = cv::Vec2f(0.0f, 0.0f);
	cube.vertex[21].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[21].texcrd = cv::Vec2f(0.0f, 1.0f);
	cube.vertex[22].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[22].texcrd = cv::Vec2f(1.0f, 1.0f);
	cube.vertex[23].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[23].texcrd = cv::Vec2f(1.0f, 0.0f);

	// the efficiency of this might be improvable...
	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 1; vi[2] = 2;
	cube.tvi.push_back(vi);
	vi[0] = 0; vi[1] = 2; vi[2] = 3;
	cube.tvi.push_back(vi);
	vi[0] = 4; vi[1] = 5; vi[2] = 6;
	cube.tvi.push_back(vi);
	vi[0] = 4; vi[1] = 6; vi[2] = 7;
	cube.tvi.push_back(vi);
	vi[0] = 8; vi[1] = 9; vi[2] = 10;
	cube.tvi.push_back(vi);
	vi[0] = 8; vi[1] = 10; vi[2] = 11;
	cube.tvi.push_back(vi);
	vi[0] = 12; vi[1] = 13; vi[2] = 14;
	cube.tvi.push_back(vi);
	vi[0] = 12; vi[1] = 14; vi[2] = 15;
	cube.tvi.push_back(vi);
	vi[0] = 16; vi[1] = 17; vi[2] = 18;
	cube.tvi.push_back(vi);
	vi[0] = 16; vi[1] = 18; vi[2] = 19;
	cube.tvi.push_back(vi);
	vi[0] = 20; vi[1] = 21; vi[2] = 22;
	cube.tvi.push_back(vi);
	vi[0] = 20; vi[1] = 22; vi[2] = 23;
	cube.tvi.push_back(vi);
	/*cube.triangleList.push_back(render::Triangle(cube.vertex[0], cube.vertex[1], cube.vertex[2]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[0], cube.vertex[2], cube.vertex[3]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[4], cube.vertex[5], cube.vertex[6]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[4], cube.vertex[6], cube.vertex[7]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[8], cube.vertex[9], cube.vertex[10]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[8], cube.vertex[10], cube.vertex[11]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[12], cube.vertex[13], cube.vertex[14]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[12], cube.vertex[14], cube.vertex[15]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[16], cube.vertex[17], cube.vertex[18]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[16], cube.vertex[18], cube.vertex[19]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[20], cube.vertex[21], cube.vertex[22]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[20], cube.vertex[22], cube.vertex[23]));*/

	cube.tci = cube.tvi;

	cube.texture = std::make_shared<Texture>();
	cube.texture->createFromFile("C:\\Users\\Patrik\\Documents\\Github\\img.png");
	cube.hasTexture = true;

	return cube;
}

Mesh MeshUtils::createPlane()
{
	Mesh plane;
	plane.vertex.resize(4);
	plane.vertex[0].position = cv::Vec4f(-0.5f, 0.0f, -0.5f, 1.0f);
	plane.vertex[1].position = cv::Vec4f(-0.5f, 0.0f, 0.5f, 1.0f);
	plane.vertex[2].position = cv::Vec4f(0.5f, 0.0f, 0.5f, 1.0f);
	plane.vertex[3].position = cv::Vec4f(0.5f, 0.0f, -0.5f, 1.0f);
	/*
	plane.vertex[0].position = cv::Vec4f(-1.0f, 1.0f, 0.0f);
	plane.vertex[1].position = cv::Vec4f(-1.0f, -1.0f, 0.0f);
	plane.vertex[2].position = cv::Vec4f(1.0f, -1.0f, 0.0f);
	plane.vertex[3].position = cv::Vec4f(1.0f, 1.0f, 0.0f);
	*/
	plane.vertex[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);
	plane.vertex[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);
	plane.vertex[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	plane.vertex[3].color = cv::Vec3f(1.0f, 1.0f, 1.0f);

	plane.vertex[0].texcrd = cv::Vec2f(0.0f, 0.0f);
	plane.vertex[1].texcrd = cv::Vec2f(0.0f, 4.0f);
	plane.vertex[2].texcrd = cv::Vec2f(4.0f, 4.0f);
	plane.vertex[3].texcrd = cv::Vec2f(4.0f, 0.0f);

	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 1; vi[2] = 2;
	plane.tvi.push_back(vi);
	vi[0] = 0; vi[1] = 2; vi[2] = 3;
	plane.tvi.push_back(vi);

	//plane.triangleList.push_back(render::Triangle(plane.vertex[0], plane.vertex[1], plane.vertex[2]));
	//plane.triangleList.push_back(render::Triangle(plane.vertex[0], plane.vertex[2], plane.vertex[3]));

	plane.texture = std::make_shared<Texture>();
	plane.texture->createFromFile("C:\\Users\\Patrik\\Cloud\\PhD\\rocks.png");
	plane.hasTexture = true;

	return plane;
}

Mesh MeshUtils::createPyramid()
{
	Mesh pyramid;
	pyramid.vertex.resize(4);

	pyramid.vertex[0].position = cv::Vec4f(-0.5f, 0.0f, 0.5f, 1.0f);
	pyramid.vertex[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);

	pyramid.vertex[1].position = cv::Vec4f(0.5f, 0.0f, 0.5f, 1.0f);
	pyramid.vertex[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);

	pyramid.vertex[2].position = cv::Vec4f(0.0f, 0.0f, -0.5f, 1.0f);
	pyramid.vertex[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	
	pyramid.vertex[3].position = cv::Vec4f(0.0f, 1.0f, 0.0f, 1.0f);
	pyramid.vertex[3].color = cv::Vec3f(0.5f, 0.5f, 0.5f);

	// the efficiency of this might be improvable...
	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 2; vi[2] = 1; // the bottom plate, draw so that visible from below
	pyramid.tvi.push_back(vi);
	vi[0] = 0; vi[1] = 1; vi[2] = 3; // front
	pyramid.tvi.push_back(vi);
	vi[0] = 2; vi[1] = 3; vi[2] = 1; // right side
	pyramid.tvi.push_back(vi);
	vi[0] = 3; vi[1] = 2; vi[2] = 0; // left side
	pyramid.tvi.push_back(vi);

	pyramid.hasTexture = false;

	return pyramid;
}

shared_ptr<Mesh> MeshUtils::createTriangle()
{
	shared_ptr<Mesh> triangle = std::make_shared<Mesh>();
	triangle->vertex.resize(3);

	triangle->vertex[0].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	triangle->vertex[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);
	triangle->vertex[0].texcrd = cv::Vec2f(0.0f, 0.0f);
	
	triangle->vertex[1].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	triangle->vertex[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);
	triangle->vertex[1].texcrd = cv::Vec2f(0.0f, 1.0f);
	
	triangle->vertex[2].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	triangle->vertex[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	triangle->vertex[2].texcrd = cv::Vec2f(1.0f, 1.0f);
	

	// the efficiency of this might be improvable...
	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 1; vi[2] = 2;
	triangle->tvi.push_back(vi);
	
	triangle->texture = std::make_shared<Texture>();
	triangle->texture->createFromFile("C:\\Users\\Patrik\\Cloud\\PhD\\up.png");
	triangle->hasTexture = false;

	return triangle;
}

	} /* namespace utils */
} /* namespace render */
