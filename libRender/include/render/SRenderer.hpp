/*!
 * \file SRenderer.h
 *
 * \author Patrik Huber
 * \date December 4, 2012
 *
 * [comment here]
 */
#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

namespace render {

#define Renderer SRenderer::Instance()

class SRenderer
{

private:
	SRenderer(void);
	~SRenderer(void);
	SRenderer(const SRenderer&);
	SRenderer& operator=(const SRenderer&);

public:
	static SRenderer* Instance(void);

public:
	void create();
	void destroy();

	void setViewport(unsigned int screenWidth, unsigned int screenHeight);

private:
	std::vector<unsigned char> colorBuffer;	// typedef unsigned char byte; Points to getSDLSurface()->pixels (void*)
	std::vector<float> depthBuffer;

	unsigned int screenWidth;
	unsigned int screenHeight;

	cv::Mat windowTransform;	// 4x4 float

};

}




