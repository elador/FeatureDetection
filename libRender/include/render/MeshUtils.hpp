/*!
 * \file MeshUtils.hpp
 *
 * \author Patrik Huber
 * \date December 12, 2012
 *
 * [comment here]
 */
#pragma once
#ifndef MESHUTILS_HPP_
#define MESHUTILS_HPP_

#include <opencv2/core/core.hpp>

#include "render/Mesh.hpp"

// Todo: Class with static methods? Or just functions? I don't know which method is better.

namespace render {

	namespace utils {

		class MeshUtils
		{
		public:
			static Mesh createCube(void);
			static Mesh createPlane(void);

			static Mesh readFromHdf5(std::string filename);

		};

	}

}
#endif /* MESHUTILS_HPP_ */
