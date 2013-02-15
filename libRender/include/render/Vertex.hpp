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

	cv::Vec4f position;	// g: f3Vec
	cv::Vec3f color;	// the color is saved as BGR!!! because opencv, the textures etc are all BGR! // g: should be fRGBA, so f4Vec.
	cv::Vec2f texcrd;	// g: f3Vec?

};

}
