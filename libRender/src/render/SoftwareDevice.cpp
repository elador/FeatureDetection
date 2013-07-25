/*
 * SoftwareDevice.cpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */

#include "render/SoftwareDevice.hpp"
#include "render/MatrixUtils.hpp"

namespace render {

SoftwareDevice::SoftwareDevice(unsigned int screenWidth, unsigned int screenHeight)
{
	setViewport(screenWidth, screenHeight);
	camera.init();
	// Todo: Set default values for the frustum? (in Camera?)

	this->colorBuffer = cv::Mat::zeros(screenHeight, screenWidth, CV_8UC4);
	this->depthBuffer = cv::Mat::ones(screenHeight, screenWidth, CV_64FC1)*1000000;

	setWorldTransform(Mat::eye(4, 4, CV_32FC1));
}

SoftwareDevice::~SoftwareDevice()
{

}

Mat SoftwareDevice::getImage()
{
	return colorBuffer;
}

Mat SoftwareDevice::getDepthBuffer()
{
	return depthBuffer;
}

void SoftwareDevice::setWorldTransform(Mat worldTransform)
{
	this->worldTransform = Mat::eye(4, 4, CV_32FC1);
}

void SoftwareDevice::updateViewTransform()
{
	Mat translate = render::utils::MatrixUtils::createTranslationMatrix(-camera.getEye()[0], -camera.getEye()[1], -camera.getEye()[2]);

	Mat rotate = (cv::Mat_<float>(4,4) << 
		camera.getRightVector()[0],		camera.getRightVector()[1],		camera.getRightVector()[2],		0.0f,
		camera.getUpVector()[0],		camera.getUpVector()[1],		camera.getUpVector()[2],		0.0f,
		-camera.getForwardVector()[0],	-camera.getForwardVector()[1],	-camera.getForwardVector()[2],	0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);

	this->viewTransform = rotate * translate;
}

void SoftwareDevice::updateProjectionTransform(bool perspective/*=true*/)
{
	Mat orthogonal = (cv::Mat_<float>(4,4) << 
		2.0f / (camera.frustum.r - camera.frustum.l),	0.0f,											0.0f,											-(camera.frustum.r + camera.frustum.l) / (camera.frustum.r - camera.frustum.l),
		0.0f,											2.0f / (camera.frustum.t - camera.frustum.b),	0.0f,											-(camera.frustum.t + camera.frustum.b) / (camera.frustum.t - camera.frustum.b),
		0.0f,											0.0f,											2.0f / (camera.frustum.n - camera.frustum.f),	-(camera.frustum.n + camera.frustum.f) / (camera.frustum.f - camera.frustum.n), // CG book has denominator (n-f) ? I had (f-n) before.
		0.0f,											0.0f,											0.0f,											1.0f);

	if (perspective) {
		Mat perspective = (cv::Mat_<float>(4,4) << 
			camera.frustum.n,	0.0f,				0.0f,									0.0f,
			0.0f,				camera.frustum.n,	0.0f,									0.0f,
			0.0f,				0.0f,				camera.frustum.n + camera.frustum.f,	+camera.frustum.n * camera.frustum.f, // CG book has -f*n ? (I had +f*n before)
			0.0f,				0.0f,				-1.0f, /* CG has +1 here, I had -1 */	0.0f);

		this->projectionTransform = orthogonal * perspective;
	} else {
		this->projectionTransform = orthogonal;
	}
}

void SoftwareDevice::setViewport(unsigned int screenWidth, unsigned int screenHeight)
{

	this->screenWidth = screenWidth;
	this->screenHeight = screenHeight;

	this->windowTransform = (cv::Mat_<float>(4,4) << 
		(float)screenWidth/2.0f,		0.0f,						0.0f,	(float)screenWidth/2.0f, // CG book says (screenWidth-1)/2.0f for second value?
		0.0f,							(float)screenHeight/2.0f,	0.0f,	(float)screenHeight/2.0f,
		0.0f,							0.0f,						1.0f,	0.0f,
		0.0f,							0.0f,						0.0f,	1.0f);

}

Vec2f SoftwareDevice::renderVertex(Vec4f vertex)
{
	// After renderVertexList is final, remove this wrapper to be more efficient.
	vector<Vec4f> list;
	list.push_back(vertex);
	vector<Vec2f> rendered = renderVertexList(list);
	return rendered[0];
}

vector<Vec2f> SoftwareDevice::renderVertexList(vector<Vec4f> vertexList)
{
	camera.update(1);
	updateViewTransform();
	updateProjectionTransform(false); // Must somewhere update at least once to be valid. in C'tor? Here is maybe not so bad?

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

} /* namespace render */
