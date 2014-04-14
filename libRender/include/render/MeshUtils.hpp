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

#include "opencv2/core/core.hpp"
#include <QtGui/QMatrix4x4>

#include <memory>

// Todo: Class with static methods? Or just functions? I don't know which method is better.

namespace render {

	namespace utils {

		class MeshUtils
		{
		public:
			static Mesh createCube();
			static Mesh createPlane();
			static Mesh createPyramid();
			static std::shared_ptr<Mesh> createTriangle();

			static cv::Mat drawTexCoords(Mesh);

			static bool isPointInTriangle(cv::Point2f point, cv::Point2f triV0, cv::Point2f triV1, cv::Point2f triV2);
			static cv::Mat extractTexture(render::Mesh mesh, QMatrix4x4 mvpMatrix, int viewportWidth, int viewportHeight, cv::Mat framebuffer); // Change QMatrix4x4 to cv::Mat so that software-renderer is not dependent on Qt?
		};

	} /* namespace utils */

} /* namespace render */

#endif /* MESHUTILS_HPP_ */
