/*
 * RenderDevice.cpp
 *
 *  Created on: 29.07.2013
 *      Author: Patrik Huber
 */

#include "render/RenderDevice.hpp"
#include "render/MatrixUtils.hpp"

using cv::Point2f;
using cv::Scalar;

namespace render {

RenderDevice::RenderDevice(unsigned int screenWidth, unsigned int screenHeight)
{
	this->screenWidth = screenWidth;
	this->screenHeight = screenHeight;
	aspect = (float)screenWidth/(float)screenHeight;

	this->colorBuffer = Mat::zeros(screenHeight, screenWidth, CV_8UC4);
	this->depthBuffer = Mat::ones(screenHeight, screenWidth, CV_64FC1)*1000000;

	setWorldTransform(Mat::eye(4, 4, CV_32FC1));
	updateViewTransform();
	updateProjectionTransform(false);
	setViewport(screenWidth, screenHeight);
}

RenderDevice::RenderDevice(unsigned int screenWidth, unsigned int screenHeight, Camera camera)
{
	this->camera = camera;

	// Todo: Use delegating c'tors as soon as VS supports it
	this->screenWidth = screenWidth;
	this->screenHeight = screenHeight;
	aspect = (float)screenWidth/(float)screenHeight;

	this->colorBuffer = Mat::zeros(screenHeight, screenWidth, CV_8UC4);
	this->depthBuffer = Mat::ones(screenHeight, screenWidth, CV_64FC1)*1000000;

	setWorldTransform(Mat::eye(4, 4, CV_32FC1));
	updateViewTransform();
	updateProjectionTransform(false);
	setViewport(screenWidth, screenHeight);
}

RenderDevice::~RenderDevice()
{
}

Mat RenderDevice::getImage()
{
	return colorBuffer;
}

Mat RenderDevice::getDepthBuffer()
{
	return depthBuffer;
}

void RenderDevice::setWorldTransform(Mat worldTransform)
{
	this->worldTransform = worldTransform;
}

void RenderDevice::updateViewTransform()
{
	Mat translate = render::utils::MatrixUtils::createTranslationMatrix(-camera.getEye()[0], -camera.getEye()[1], -camera.getEye()[2]);
	
	Mat rotate = (cv::Mat_<float>(4,4) << 
		camera.getRightVector()[0],		camera.getRightVector()[1],		camera.getRightVector()[2],		0.0f,
		camera.getUpVector()[0],		camera.getUpVector()[1],		camera.getUpVector()[2],		0.0f,
		camera.getForwardVector()[0],	camera.getForwardVector()[1],	camera.getForwardVector()[2],	0.0f,
		0.0f,							0.0f,							0.0f,							1.0f);
	viewTransform = rotate * translate;
}


void RenderDevice::updateProjectionTransform(bool perspective/*=true*/)
{
	Mat orthogonal = (cv::Mat_<float>(4,4) << 
		2.0f / (camera.frustum.r - camera.frustum.l),	0.0f,											0.0f,											-(camera.frustum.r + camera.frustum.l) / (camera.frustum.r - camera.frustum.l),
		0.0f,											2.0f / (camera.frustum.t - camera.frustum.b),	0.0f,											-(camera.frustum.t + camera.frustum.b) / (camera.frustum.t - camera.frustum.b),
		0.0f,											0.0f,											2.0f / (camera.frustum.n - camera.frustum.f),	-(camera.frustum.n + camera.frustum.f) / (camera.frustum.n - camera.frustum.f), // CG book has denominator (n-f) ? I had (f-n) before. When n and f are neg and here is n-f, then it's the same as n and f pos and f-n here.
		0.0f,											0.0f,											0.0f,											1.0f);
	if (perspective) {
		Mat perspective = (cv::Mat_<float>(4,4) << 
			camera.frustum.n,	0.0f,				0.0f,									0.0f,
			0.0f,				camera.frustum.n,	0.0f,									0.0f,
			0.0f,				0.0f,				camera.frustum.n + camera.frustum.f,	-camera.frustum.n * camera.frustum.f, // CG book has -f*n ? (I had +f*n before). (doesn't matter, cancels when either both n and f neg or both pos)
			0.0f,				0.0f,				+1.0f, /* CG has +1 here, I had -1 */	0.0f);
		projectionTransform = orthogonal * perspective;
	} else {
		projectionTransform = orthogonal;
	}
}

void RenderDevice::setViewport(unsigned int screenWidth, unsigned int screenHeight)
{
	windowTransform = (cv::Mat_<float>(4,4) << 
		(float)screenWidth/2.0f,		0.0f,						0.0f,	(float)screenWidth/2.0f, // CG book says (screenWidth-1)/2.0f for second value?
		0.0f,							-(float)screenHeight/2.0f,	0.0f,	(float)screenHeight/2.0f,
		0.0f,							0.0f,						1.0f,	0.0f,
		0.0f,							0.0f,						0.0f,	1.0f);
}

Vec2f RenderDevice::projectVertex(Vec4f vertex)
{
	// After renderVertexList is final, remove this wrapper to be more efficient.
	vector<Vec4f> list;
	list.push_back(vertex);
	vector<Vec2f> rendered = projectVertexList(list);
	return rendered[0];
}

vector<Vec2f> RenderDevice::projectVertexList(vector<Vec4f> vertexList)
{
	vector<Vec2f> pointList;

	for (const auto& vertex : vertexList) {		// Note: We could put all the points in a matrix and then transform only this matrix?
		Mat worldSpace = worldTransform * Mat(vertex);
		Mat camSpace = viewTransform * worldSpace;
		Mat normalizedViewingVolume = projectionTransform * camSpace;
		
		// project from 4D to 2D window position with depth value in z coordinate
		Vec4f normViewVolVec = matToColVec4f(normalizedViewingVolume);
		normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
		Mat windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
		Vec4f windowCoordsVec = matToColVec4f(windowCoords);

		Vec2f p = Vec2f(windowCoordsVec[0], windowCoordsVec[1]);
		pointList.push_back(p);
	}

	return pointList;
}

void RenderDevice::resetBuffers()
{
	colorBuffer = Mat::zeros(screenHeight, screenWidth, CV_8UC4);
	depthBuffer = Mat::ones(screenHeight, screenWidth, CV_64FC1)*1000000;
}

void RenderDevice::renderLine(Vec4f p0, Vec4f p1, Scalar color)
{
	Mat worldSpace = worldTransform * Mat(p0);
	Mat camSpace = viewTransform * worldSpace;
	Mat normalizedViewingVolume = projectionTransform * camSpace;
	Vec4f normViewVolVec = matToColVec4f(normalizedViewingVolume);
	normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
	Mat windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
	Vec4f windowCoordsVec = matToColVec4f(windowCoords);
	Point2f p0Screen(windowCoordsVec[0], windowCoordsVec[1]);

	worldSpace = worldTransform * Mat(p1);
	camSpace = viewTransform * worldSpace;
	normalizedViewingVolume = projectionTransform * camSpace;
	normViewVolVec = matToColVec4f(normalizedViewingVolume);
	normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
	windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
	windowCoordsVec = matToColVec4f(windowCoords);
	Point2f p1Screen(windowCoordsVec[0], windowCoordsVec[1]);

	line(colorBuffer, p0Screen, p1Screen, color);
}

} /* namespace render */
