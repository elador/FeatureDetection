/*
 * MeshUtils.hpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef MESHUTILS_HPP_
#define MESHUTILS_HPP_

#include "render2/Mesh.hpp"
#include "render2/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

// Todo: Class with static methods? Or just functions? I don't know which method is better.

namespace render {

	namespace utils {

		class MeshUtils
		{
		public:
			static Mesh createCube(void);
			static Mesh createPlane(void);

			static Mesh readFromHdf5(std::string filename);
			static MorphableModel readFromScm(std::string filename);

		};

	} /* namespace utils */

} /* namespace render */

#endif /* MESHUTILS_HPP_ */
