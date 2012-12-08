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

#include "render/Triangle.hpp"

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

	cv::Mat constructViewTransform(const cv::Vec3f& position, const cv::Vec3f& rightVector, const cv::Vec3f& upVector, const cv::Vec3f& forwardVector);
	cv::Mat constructProjTransform(float left, float right, float bottom, float top, float zNear, float zFar);

	void setTexture(const Texture& texture);
	void setTrianglesBuffer(const std::vector<Triangle> trianglesBuffer);
	void setTransform(const cv::Mat transform);
	void draw(ushort trianglesNum = 0);
	void end();

private:

	struct DrawCall	{
		std::vector<Triangle> trianglesBuffer;
		unsigned int trianglesNum;
		cv::Mat transform;
		const Texture* texture;
	};

	//std::vector<unsigned char> colorBuffer;	// typedef unsigned char byte; Points to getSDLSurface()->pixels (void*)
	cv::Mat colorBuffer;
	//std::vector<float> depthBuffer;
	cv::Mat depthBuffer;

	unsigned int screenWidth;
	unsigned int screenHeight;
	int screenWidth_tiles;
	int screenHeight_tiles;

	cv::Mat windowTransform;	// 4x4 float

	std::vector<Triangle> currentTrianglesBuffer;
	cv::Mat currentTransform;

	std::vector<DrawCall> drawCalls;
	std::vector<TriangleToRasterize> trianglesToRasterize; // holds copies of all triangles from called triangles buffers that are to be rendered so the pipeline can work on them instead of on the original triangles

	const Texture* currentTexture;

	void runVertexProcessor();
	Vertex runVertexShader(const cv::Mat& transform, const Vertex& input);
	void processProspectiveTriangleToRasterize(const Vertex& _v0, const Vertex& _v1, const Vertex& _v2, const Texture* _texture);
	std::vector<Vertex> clipPolygonToPlaneIn4D(const std::vector<Vertex>& vertices, const cv::Vec4f& planeNormal);

	bool areVerticesCCWInScreenSpace(const Vertex& v0, const Vertex& v1, const Vertex& v2);	// should better go to a RenderUtils class?
	float implicitLine(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2);	// ->utils ?


	/* Pixel processing: */
	void runPixelProcessor();

	float dudx, dudy, dvdx, dvdy; // partial derivatives of U/V coordinates with respect to X/Y pixel's screen coordinates
	
	cv::Vec3f runPixelShader(const Texture* texture, const cv::Vec3f& color, const cv::Vec2f& texCoord);
	cv::Vec3f tex2D(const Texture* texture, const cv::Vec2f& texCoord);
	cv::Vec3f tex2D_linear_mipmap_linear(const Texture* texture, const cv::Vec2f& texCoord);
	cv::Vec3f tex2D_linear(const Texture* texture, const cv::Vec2f& imageTexCoord, unsigned char mipmapIndex);
	cv::Vec2f texCoord_wrap(const cv::Vec2f& texCoord);
	float clamp(float x, float a, float b);
};

}




