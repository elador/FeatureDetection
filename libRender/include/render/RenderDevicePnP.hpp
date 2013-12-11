/*
 * RenderDevicePnP.hpp
 *
 *  Created on: 25.11.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef RENDERDEVICEPNP_HPP_
#define RENDERDEVICEPNP_HPP_

#include "render/Camera.hpp"
#include "render/Mesh.hpp"

#include "opencv2/core/core.hpp"

#include <vector>
#include <memory>

using cv::Mat;
using cv::Vec2f;
using cv::Vec4f;
using cv::Scalar;
using std::vector;
using std::shared_ptr;

namespace render {

/**
 * Desc
 */
class RenderDevicePnP
{
public:
	enum class RenderMode {
		WIREFRAME, ///< Todo
		VERTEX_COLORING,   ///< Todo
		TEXTURING   ///< Todo
	};


	// Future Todo: Use c++11 delegating c'tors as soon as VS supports it, i.e. create a viewport with 640x480 default
	
	RenderDevicePnP(unsigned int screenWidth, unsigned int screenHeight);
	RenderDevicePnP(unsigned int screenWidth, unsigned int screenHeight, Camera camera);
	~RenderDevicePnP(); // Why no virtual possible?

	Mat getImage(); // make these a '&' ?
	Mat getDepthBuffer();

	Camera& getCamera() {
			return camera;
	};

	void setBackgroundImage(Mat background);

	void updateWindow();

	// Render... Does not do any clipping.
	Vec2f projectVertex(Vec4f vertex);
	vector<Vec2f> projectVertexList(vector<Vec4f> vertexList);
	std::pair<Vec2f, bool> projectVertexVis(Vec4f vertex);
	void renderLine(Vec4f p0, Vec4f p1, Scalar color);
	void renderLM(Vec4f p0, Scalar color);
	void renderMesh(Mesh mesh);

	void setModelTransform(Mat modelTransform);

	void setExtrinsicCameraTransform(Mat extrinsicCameraTransform) {
		this->extrinsicCameraTransform = extrinsicCameraTransform;
	}
	void setIntrinsicCameraTransform(Mat intrinsicCameraTransform) {
		this->intrinsicCameraTransform = intrinsicCameraTransform;	
	};

	void resetBuffers();

	// Todo: move the camera to Renderer, not RenderDevice.
	Camera camera;
	void updateViewTransform();
	void updateProjectionTransform(bool perspective=true);

	void draw(shared_ptr<Mesh> mesh, shared_ptr<Texture> texture);

protected:
	Mat colorBuffer;
	Mat depthBuffer;
	unsigned int screenWidth;
	unsigned int screenHeight;
	float aspect;
	

	Mat worldTransform;			// Model-matrix. We should have/save one per object if we start rendering more than 1 object.
	Mat viewTransform;			// Camera-transform
	Mat projectionTransform;	// Orthogonal or projective transform
	Mat windowTransform;	// Transform to window coordinates, 4 x 4 float

	Mat extrinsicCameraTransform; // From PnP algos
	Mat intrinsicCameraTransform; // From PnP algos

	void setViewport(unsigned int screenWidth, unsigned int screenHeight);


	// "old" rasterizing functions:
	struct DrawCall	{
		shared_ptr<Mesh> mesh; //std::vector<std::tuple<int, int, int>> triangleIndicesBuffer;
		unsigned int trianglesNum;
		cv::Mat transform;
		shared_ptr<Texture> texture;
	};
	std::vector<DrawCall> drawCalls;
	std::vector<TriangleToRasterize> trianglesToRasterize; // holds copies of all triangles from called triangles buffers that are to be rendered so the pipeline can work on them instead of on the original triangles

	void runVertexProcessor();
	Vertex runVertexShader(shared_ptr<Mesh> mesh, const cv::Mat& transform, const int vertexNum);
	void processProspectiveTriangleToRasterize(const Vertex& _v0, const Vertex& _v1, const Vertex& _v2, shared_ptr<Texture> _texture);
	std::vector<Vertex> clipPolygonToPlaneIn4D(const std::vector<Vertex>& vertices, const cv::Vec4f& planeNormal);

	bool areVerticesCCWInScreenSpace(const Vertex& v0, const Vertex& v1, const Vertex& v2);	// should better go to a RenderUtils class?
	double implicitLine(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2);	// ->utils ?

	/* Pixel processing: */
	void runPixelProcessor();

	float dudx, dudy, dvdx, dvdy; // partial derivatives of U/V coordinates with respect to X/Y pixel's screen coordinates

	cv::Vec3f runPixelShader(shared_ptr<Texture> texture, const cv::Vec3f& color, const cv::Vec2f& texCoord, bool useTexturing=true);
	cv::Vec3f tex2D(shared_ptr<Texture> texture, const cv::Vec2f& texCoord);
	cv::Vec3f tex2D_linear_mipmap_linear(shared_ptr<Texture> texture, const cv::Vec2f& texCoord);
	cv::Vec3f tex2D_linear(shared_ptr<Texture> texture, const cv::Vec2f& imageTexCoord, unsigned char mipmapIndex);
	cv::Vec2f texCoord_wrap(const cv::Vec2f& texCoord);
	float clamp(float x, float a, float b);
	// END "old" rasterizing functions

	cv::Vec4f matToColVec4f(cv::Mat m) {
		cv::Vec4f ret;
		ret[0] = m.at<float>(0, 0);
		ret[1] = m.at<float>(1, 0);
		ret[2] = m.at<float>(2, 0);
		ret[3] = m.at<float>(3, 0);
		return ret;
	}
};

 } /* namespace render */

#endif /* RENDERDEVICEPNP_HPP_ */
