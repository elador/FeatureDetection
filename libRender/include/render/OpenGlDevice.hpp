/*
 * OpenGlDevice.hpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef OPENGLDEVICE_HPP_
#define OPENGLDEVICE_HPP_

#include "render/RenderDevice.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/core/opengl_interop.hpp"

#include <string>
#include <memory>

using std::string;
using std::shared_ptr;

namespace render {

/**
 * Desc
 */
class OpenGlDevice : public RenderDevice
{

public:
	OpenGlDevice() {};
	OpenGlDevice(unsigned int screenWidth, unsigned int screenHeight);
	~OpenGlDevice();

	cv::Mat getImage();
	cv::Mat getDepthBuffer();
	
	Vec2f renderVertex(Vec4f vertex);
	vector<Vec2f> renderVertexList(vector<Vec4f> vertexList);
	void renderMesh(Mesh mesh);

	void setWorldTransform(Mat worldTransform);

	void setBackgroundImage(Mat background);

	void updateWindow();
	
private:
	enum class RenderTypeState { NONE, VERTEX, VERTEXLIST, MESH };
	RenderTypeState renderTypeState;

	cv::Mat colorBuffer;
	cv::Mat depthBuffer;

	string windowName;
	static void openGlDrawCallback(void* userdata);
	void openGlDrawCallbackReal();

	cv::GlTexture2D backgroundTex; // shared_ptr?

	void drawAxes(float scale=1); // 1 = unit axes
	void drawBackground();
	void drawPlaneXY(float x, float y, float z, float scale=1);
	void drawCube(float x, float y, float z, float scale=1);

	void drawMesh();
	Mesh meshToDraw; // 1) make shared_ptr. 2) Problems as soon as we want to draw more than 1 object
	void drawVertex(Vertex v);
};

} /* namespace render */

#endif /* OPENGLDEVICE_HPP_ */
