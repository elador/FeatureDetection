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
	Vertex(void);
	Vertex::Vertex(const cv::Vec4f& position, const cv::Vec3f& color, const cv::Vec2f& texCoord);
	~Vertex(void);

	cv::Vec4f position;	// g: f3Vec
	cv::Vec3f color;	// the color is saved as BGR!!! because opencv, the textures etc are all BGR! // g: should be fRGBA, so f4Vec.
	cv::Vec2f texcrd;	// g: f3Vec?

};

} /* namespace render */

#endif /* VERTEX_HPP_ */
