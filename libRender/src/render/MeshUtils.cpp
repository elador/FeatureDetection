/*!
 * \file MeshUtils.cpp
 *
 * \author Patrik Huber
 * \date December 12, 2012
 *
 * [comment here]
 */

#include "render/MeshUtils.hpp"

namespace render {
	namespace utils {

Mesh MeshUtils::createCube(void)
{
	Mesh cube;
	cube.vertices.resize(24);

	for (int i = 0; i < 24; i++)
		cube.vertices[i].color = cv::Vec3f(1.0f, 1.0f, 1.0f);

	cube.vertices[0].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertices[0].texCoord = cv::Vec2f(0.0f, 0.0f);
	cube.vertices[1].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertices[1].texCoord = cv::Vec2f(0.0f, 1.0f);
	cube.vertices[2].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertices[2].texCoord = cv::Vec2f(1.0f, 1.0f);
	cube.vertices[3].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertices[3].texCoord = cv::Vec2f(1.0f, 0.0f);

	cube.vertices[4].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertices[4].texCoord = cv::Vec2f(0.0f, 0.0f);
	cube.vertices[5].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertices[5].texCoord = cv::Vec2f(0.0f, 1.0f);
	cube.vertices[6].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertices[6].texCoord = cv::Vec2f(1.0f, 1.0f);
	cube.vertices[7].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertices[7].texCoord = cv::Vec2f(1.0f, 0.0f);

	cube.vertices[8].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertices[8].texCoord = cv::Vec2f(0.0f, 0.0f);
	cube.vertices[9].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertices[9].texCoord = cv::Vec2f(0.0f, 1.0f);
	cube.vertices[10].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertices[10].texCoord = cv::Vec2f(1.0f, 1.0f);
	cube.vertices[11].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertices[11].texCoord = cv::Vec2f(1.0f, 0.0f);

	cube.vertices[12].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertices[12].texCoord = cv::Vec2f(0.0f, 0.0f);
	cube.vertices[13].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertices[13].texCoord = cv::Vec2f(0.0f, 1.0f);
	cube.vertices[14].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertices[14].texCoord = cv::Vec2f(1.0f, 1.0f);
	cube.vertices[15].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertices[15].texCoord = cv::Vec2f(1.0f, 0.0f);

	cube.vertices[16].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertices[16].texCoord = cv::Vec2f(0.0f, 0.0f);
	cube.vertices[17].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertices[17].texCoord = cv::Vec2f(0.0f, 1.0f);
	cube.vertices[18].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertices[18].texCoord = cv::Vec2f(1.0f, 1.0f);
	cube.vertices[19].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertices[19].texCoord = cv::Vec2f(1.0f, 0.0f);

	cube.vertices[20].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertices[20].texCoord = cv::Vec2f(0.0f, 0.0f);
	cube.vertices[21].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertices[21].texCoord = cv::Vec2f(0.0f, 1.0f);
	cube.vertices[22].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertices[22].texCoord = cv::Vec2f(1.0f, 1.0f);
	cube.vertices[23].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertices[23].texCoord = cv::Vec2f(1.0f, 0.0f);

	// the efficiency of this might be improvable
	cube.triangleList.push_back(render::Triangle(cube.vertices[0], cube.vertices[1], cube.vertices[2]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[0], cube.vertices[2], cube.vertices[3]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[4], cube.vertices[5], cube.vertices[6]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[4], cube.vertices[6], cube.vertices[7]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[8], cube.vertices[9], cube.vertices[10]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[8], cube.vertices[10], cube.vertices[11]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[12], cube.vertices[13], cube.vertices[14]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[12], cube.vertices[14], cube.vertices[15]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[16], cube.vertices[17], cube.vertices[18]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[16], cube.vertices[18], cube.vertices[19]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[20], cube.vertices[21], cube.vertices[22]));
	cube.triangleList.push_back(render::Triangle(cube.vertices[20], cube.vertices[22], cube.vertices[23]));

	cube.texture.createFromFile("data/pwr.png");

	return cube;
}

Mesh MeshUtils::createPlane(void)
{
	Mesh plane;
	plane.vertices.resize(4);
	plane.vertices[0].position = cv::Vec4f(-0.5f, 0.0f, -0.5f, 1.0f);
	plane.vertices[1].position = cv::Vec4f(-0.5f, 0.0f, 0.5f, 1.0f);
	plane.vertices[2].position = cv::Vec4f(0.5f, 0.0f, 0.5f, 1.0f);
	plane.vertices[3].position = cv::Vec4f(0.5f, 0.0f, -0.5f, 1.0f);
	/*
	plane.vertices[0].position = cv::Vec4f(-1.0f, 1.0f, 0.0f);
	plane.vertices[1].position = cv::Vec4f(-1.0f, -1.0f, 0.0f);
	plane.vertices[2].position = cv::Vec4f(1.0f, -1.0f, 0.0f);
	plane.vertices[3].position = cv::Vec4f(1.0f, 1.0f, 0.0f);
	*/
	plane.vertices[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);
	plane.vertices[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);
	plane.vertices[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	plane.vertices[3].color = cv::Vec3f(1.0f, 1.0f, 1.0f);

	plane.vertices[0].texCoord = cv::Vec2f(0.0f, 0.0f);
	plane.vertices[1].texCoord = cv::Vec2f(0.0f, 4.0f);
	plane.vertices[2].texCoord = cv::Vec2f(4.0f, 4.0f);
	plane.vertices[3].texCoord = cv::Vec2f(4.0f, 0.0f);

	plane.triangleList.push_back(render::Triangle(plane.vertices[0], plane.vertices[1], plane.vertices[2]));
	plane.triangleList.push_back(render::Triangle(plane.vertices[0], plane.vertices[2], plane.vertices[3]));

	plane.texture.createFromFile("data/rocks.png");

	return plane;
}

Mesh MeshUtils::readFromHdf5(std::string filename)
{
	Mesh tmp;
	return tmp;
}

	} /* END namespace utils */
} /* END namespace render */
