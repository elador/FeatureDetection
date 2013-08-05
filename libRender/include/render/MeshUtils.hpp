/*
 * MeshUtils.hpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef MESHUTILS_HPP_
#define MESHUTILS_HPP_

#include "render/Mesh.hpp"
#include "render/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include <memory>

using std::shared_ptr;

// Todo: Class with static methods? Or just functions? I don't know which method is better.

namespace render {

	namespace utils {

		class MeshUtils
		{
		public:
			static Mesh createCube();
			static Mesh createPlane();
			static Mesh createPyramid();
			static shared_ptr<Mesh> createTriangle();

			static Mesh readFromHdf5(std::string filename);
			static MorphableModel readFromScm(std::string filename);

		};

	} /* namespace utils */

} /* namespace render */

#endif /* MESHUTILS_HPP_ */
