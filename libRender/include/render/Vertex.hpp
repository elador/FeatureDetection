/*!
 * \file Vertex.h
 *
 * \author Patrik Huber
 * \date December 4, 2012
 *
 * [comment here]
 */
#pragma once

#include <opencv2/core/core.hpp>

namespace render {

class Vertex
{
public:
	Vertex(void);
	Vertex::Vertex(const cv::Vec4f& position, const cv::Vec3f& color, const cv::Vec2f& texCoord);
	~Vertex(void);

	cv::Vec4f position;
	cv::Vec3f color;
	cv::Vec2f texCoord;

};

}
