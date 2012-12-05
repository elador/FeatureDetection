/*!
 * \file SRenderer.cpp
 *
 * \author Patrik Huber
 * \date December 4, 2012
 *
 * [comment here]
 */

#include "render/SRenderer.hpp"

namespace render {

SRenderer::SRenderer(void)
{
}


SRenderer::~SRenderer(void)
{
}

SRenderer* SRenderer::Instance(void)
{
	static SRenderer instance;
	return &instance;
}


void SRenderer::create()
{
	setViewport(640, 480);

	//screenWidth_tiles = (screenWidth + 15) / 16;
	//screenHeight_tiles = (screenHeight + 15) / 16;

	colorBuffer.resize(screenWidth * screenHeight);	// or .reserve?
	depthBuffer.resize(screenWidth * screenHeight);	// or .reserve?

}

void SRenderer::destroy()
{
}

void SRenderer::setViewport(unsigned int screenWidth, unsigned int screenHeight)
{
	this->screenWidth = screenWidth;
	this->screenHeight = screenHeight;

	this->windowTransform = (cv::Mat_<float>(4,4) << 
						  (float)screenWidth/2.0f,		 0.0f,						0.0f,	0.0f,
						  0.0f,							 (float)screenHeight/2.0f,	0.0f,	0.0f,
						  0.0f,							 0.0f,						1.0f,	0.0f,
						  (float)screenWidth/2.0f,		 (float)screenHeight/2.0f,	0.0f,	1.0f);
	// order: M(row, col);

}

}