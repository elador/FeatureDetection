/*
 * SoftwareDevice.cpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */

#include "render/SoftwareDevice.hpp"
#include "render/MatrixUtils.hpp"

namespace render {
/*
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
	this->worldTransform = worldTransform;
}



void SoftwareDevice::updateProjectionTransform(bool perspective/*=true*//*)
{

}

void SoftwareDevice::setViewport(unsigned int screenWidth, unsigned int screenHeight)
{

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
*/
} /* namespace render */
