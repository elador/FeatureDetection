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

using std::string;

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

	void setWorldTransform(Mat worldTransform);

	void setBackgroundImage(Mat background);

	void updateWindow();
	
private:
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

};

} /* namespace render */

#endif /* OPENGLDEVICE_HPP_ */
