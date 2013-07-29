/*
 * SoftwareDevice.hpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef SOFTWAREDEVICE_HPP_
#define SOFTWAREDEVICE_HPP_

#include "render/RenderDevice.hpp"

namespace render {

/**
 * Desc
 */
/*
class SoftwareDevice : public RenderDevice
{

public:
	SoftwareDevice() {};
	SoftwareDevice(unsigned int screenWidth, unsigned int screenHeight);
	~SoftwareDevice();

	Mat getImage();		// really necessary? include in functions below?
	Mat getDepthBuffer();

	// Render... Does not do any clipping.
	Vec2f renderVertex(Vec4f vertex);
	vector<Vec2f> renderVertexList(vector<Vec4f> vertexList);

	//Mat renderMesh(Mesh mesh, vector<int> mask=vector<int>());

	void setWorldTransform(Mat worldTransform);

	void setBackgroundImage(Mat background) {};

private:
	Mat colorBuffer;
	Mat depthBuffer;

	Mat worldTransform;			// Model-matrix. We should have/save one per object if we start rendering more than 1 object.
	Mat viewTransform;			// Camera-transform
	Mat projectionTransform;	// Orthogonal or projective transform
	Mat windowTransform;	// Transform to window coordinates, 4 x 4 float

	
	void updateViewTransform();
	void updateProjectionTransform(bool perspective=true);
	void setViewport(unsigned int screenWidth, unsigned int screenHeight);

	// Helper
	cv::Vec4f matToColVec4f(cv::Mat m) {
		cv::Vec4f ret;
		ret[0] = m.at<float>(0, 0);
		ret[1] = m.at<float>(1, 0);
		ret[2] = m.at<float>(2, 0);
		ret[3] = m.at<float>(3, 0);
		return ret;
	};
};
*/
 } /* namespace render */

#endif /* SOFTWAREDEVICE_HPP_ */
