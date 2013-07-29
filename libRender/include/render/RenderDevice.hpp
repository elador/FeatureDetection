/*
 * RenderDevice.hpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef RENDERDEVICE_HPP_
#define RENDERDEVICE_HPP_

#include "render/Camera.hpp"
#include "render/Mesh.hpp"

#include "opencv2/core/core.hpp"

#include <vector>

using cv::Mat;
using cv::Vec2f;
using cv::Vec4f;
using std::vector;

namespace render {

/**
 * Desc
 */
class RenderDevice
{
public:
	RenderDevice(); // I think I don't want a default constructor?
	RenderDevice(unsigned int screenWidth, unsigned int screenHeight);
	~RenderDevice(); // Why no virtual possible?

	cv::Mat getImage(); // make these a '&' ?
	cv::Mat getDepthBuffer();

	Camera& getCamera() {
			return camera;
	};

	void setBackgroundImage(Mat background);

	void updateWindow();

	// Render... Does not do any clipping.
	Vec2f projectVertex(Vec4f vertex);
	vector<Vec2f> projectVertexList(vector<Vec4f> vertexList);
	void renderMesh(Mesh mesh);

	void setWorldTransform(Mat worldTransform);

	void resetBuffers();

protected:
	Mat colorBuffer;
	Mat depthBuffer;
	unsigned int screenWidth;
	unsigned int screenHeight;
	float aspect;
	Camera camera; // init this in c'tor? (no it's on stack so default Camera c'tor init)

	Mat worldTransform;			// Model-matrix. We should have/save one per object if we start rendering more than 1 object.
	Mat viewTransform;			// Camera-transform
	Mat projectionTransform;	// Orthogonal or projective transform
	Mat windowTransform;	// Transform to window coordinates, 4 x 4 float


	void updateViewTransform();
	void updateProjectionTransform(bool perspective=true);
	void setViewport(unsigned int screenWidth, unsigned int screenHeight);

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

#endif /* RENDERDEVICE_HPP_ */
