/*!
 * \file SRenderer.cpp
 *
 * \author Patrik Huber
 * \date December 4, 2012
 *
 * [comment here]
 */

#include "render/SRenderer.hpp"
#include "render/MatrixUtils.hpp"

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

cv::Mat SRenderer::constructViewTransform(const cv::Vec3f& position, const cv::Vec3f& rightVector, const cv::Vec3f& upVector, const cv::Vec3f& forwardVector)
{
	cv::Mat translate = render::utils::MatrixUtils::createTranslationMatrix(-position[0], -position[1], -position[2]);

	cv::Mat rotate = (cv::Mat_<float>(4,4) << 
				   rightVector[0],	upVector[0],		forwardVector[0],	0.0f,
			       rightVector[1],	upVector[1],		forwardVector[1],	0.0f,
			       rightVector[2],	upVector[2],		forwardVector[2],	0.0f,
			       0.0f,			0.0f,				0.0f,				1.0f);

	return translate * rotate;
}

cv::Mat SRenderer::constructProjTransform(float left, float right, float bottom, float top, float zNear, float zFar)
{
	cv::Mat perspective = (cv::Mat_<float>(4,4) << 
						zNear,	0.0f,	0.0f,			0.0f,
						0.0f,	zNear,	0.0f,			0.0f,
						0.0f,	0.0f,	zNear + zFar,	-1.0f,
						0.0f,	0.0f,	zNear * zFar,	0.0f);

	cv::Mat orthogonal = (cv::Mat_<float>(4,4) << 
					   2.0f / (right - left),				0.0f,								0.0f,								0.0f,
					   0.0f,								2.0f / (top - bottom),				0.0f,								0.0f,
					   0.0f,								0.0f,								-2.0f / (zFar - zNear),				0.0f,
					   -(right + left) / (right - left),	-(top + bottom) / (top - bottom),	-(zNear + zFar) / (zFar - zNear),	1.0f);
	
	return perspective * orthogonal;
}

void SRenderer::setTrianglesBuffer(const std::vector<Triangle> trianglesBuffer)
{
	currentTrianglesBuffer = trianglesBuffer;	// maybe "&trianglesBuffer" for better speed? how does std::vector behave?
}

void SRenderer::setTransform(const cv::Mat transform)
{
	currentTransform = transform;
}

void SRenderer::draw(ushort trianglesNum)
{
	DrawCall drawCall;

	drawCall.trianglesBuffer = currentTrianglesBuffer;
	drawCall.trianglesNum = (trianglesNum == 0 ? currentTrianglesBuffer.size() : trianglesNum);
	drawCall.transform = currentTransform;
	//drawCall.texture = currentTexture;

	drawCalls.push_back(drawCall);
}

void SRenderer::end()
{
	//runVertexProcessor();
	//runPixelProcessor();

	drawCalls.clear();
	trianglesToRasterize.clear();


	//screenSurface;	// draw on screen, flip buffer (with double-buffering) or just draw the image
	// probably write the colorBuffer to a .png.
	
	
}

}