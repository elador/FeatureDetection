/*
 * Vertex.hpp
 *
 *  Created on: 04.12.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef VERTEX_HPP_
#define VERTEX_HPP_

#include "opencv2/core/core.hpp"

namespace render {

/**
 * Desc
 */
class Vertex
{
public:
	Vertex();
	Vertex(const cv::Vec4f& position, const cv::Vec3f& color, const cv::Vec2f& texCoord);

	cv::Vec4f position;	///< doc.
	cv::Vec3f color;	///< RGB-format
	cv::Vec2f texcrd;	///< doc.
};

} /* namespace render */

#endif /* VERTEX_HPP_ */
