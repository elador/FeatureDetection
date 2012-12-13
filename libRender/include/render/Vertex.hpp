/*!
 * \file Vertex.hpp
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

	cv::Vec4f position;	// f3Vec
	cv::Vec3f color;	// should be fRGBA, so f4Vec
	cv::Vec2f texCoord;	// f3Vec?

};

}
