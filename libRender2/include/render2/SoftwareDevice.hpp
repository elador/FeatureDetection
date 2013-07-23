/*
 * SoftwareDevice.hpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef SOFTWAREDEVICE_HPP_
#define SOFTWAREDEVICE_HPP_

#include "render2/Triangle.hpp"
#include "render2/Mesh.hpp"
#include "render2/Camera.hpp"

#include "opencv2/core/core.hpp"

#include <vector>

namespace render {

#define Renderer SRenderer::Instance()

/**
 * Desc
 */
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

	cv::Mat constructViewTransform();
	cv::Mat constructProjTransform();

	void setTexture(const Texture& texture);
	void setMesh(const Mesh* mesh);
	void setTransform(const cv::Mat transform);
	void draw(ushort trianglesNum = 0);
	void end();

	cv::Mat getRendererImage();
	cv::Mat getRendererDepthBuffer();

	cv::Mat getWindowTransform() {
		return windowTransform;
	}

	Camera camera;	// TODO make private

private:

	struct DrawCall	{
		const Mesh* mesh; //std::vector<std::tuple<int, int, int>> triangleIndicesBuffer;
		unsigned int trianglesNum;
		cv::Mat transform;
		const Texture* texture;
	};

	cv::Mat colorBuffer;
	cv::Mat depthBuffer;

	unsigned int screenWidth;
	unsigned int screenHeight;

	cv::Mat windowTransform;	// 4x4 float

	const Mesh* currentMesh;	// we are not allowed to change the mesh
	cv::Mat currentTransform;

	std::vector<DrawCall> drawCalls;
	std::vector<TriangleToRasterize> trianglesToRasterize; // holds copies of all triangles from called triangles buffers that are to be rendered so the pipeline can work on them instead of on the original triangles

	const Texture* currentTexture;

	void runVertexProcessor();
	Vertex runVertexShader(const Mesh* mesh, const cv::Mat& transform, const int vertexNum);
	void processProspectiveTriangleToRasterize(const Vertex& _v0, const Vertex& _v1, const Vertex& _v2, const Texture* _texture);
	std::vector<Vertex> clipPolygonToPlaneIn4D(const std::vector<Vertex>& vertices, const cv::Vec4f& planeNormal);

	bool areVerticesCCWInScreenSpace(const Vertex& v0, const Vertex& v1, const Vertex& v2);	// should better go to a RenderUtils class?
	double implicitLine(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2);	// ->utils ?

	/* Pixel processing: */
	void runPixelProcessor();

	float dudx, dudy, dvdx, dvdy; // partial derivatives of U/V coordinates with respect to X/Y pixel's screen coordinates
	
	cv::Vec3f runPixelShader(const Texture* texture, const cv::Vec3f& color, const cv::Vec2f& texCoord, bool useTexturing=true);
	cv::Vec3f tex2D(const Texture* texture, const cv::Vec2f& texCoord);
	cv::Vec3f tex2D_linear_mipmap_linear(const Texture* texture, const cv::Vec2f& texCoord);
	cv::Vec3f tex2D_linear(const Texture* texture, const cv::Vec2f& imageTexCoord, unsigned char mipmapIndex);
	cv::Vec2f texCoord_wrap(const cv::Vec2f& texCoord);
	float clamp(float x, float a, float b);
};

 } /* namespace render */

#endif /* SOFTWAREDEVICE_HPP_ */
