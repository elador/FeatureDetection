/*
 * Mesh.cpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */

#include "render/Mesh.hpp"

#include <fstream>

using std::string;

namespace render {

Mesh::Mesh()
{
	hasTexture = false;
}


Mesh::~Mesh()
{
}

void Mesh::writeObj(Mesh mesh, string filename)
{
	std::ofstream objFile(filename);

	for (const auto& v : mesh.vertex) {
		objFile << "v " << v.position[0] << " " << v.position[1] << " " << v.position[2] << std::endl;
	}

	for (const auto& v : mesh.tvi) {
		objFile << "f " << v[0]+1 << " " << v[1]+1 << " " << v[2]+1 << std::endl;
	}
	objFile.close();
	return;

	// obj starts counting triangles at 1
}

} /* namespace render */
