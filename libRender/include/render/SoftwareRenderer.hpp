/*
 * SoftwareRenderer.hpp
 *
 *  Created on: 06.04.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef SOFTWARERENDERER_HPP_
#define SOFTWARERENDERER_HPP_

#include "render/Mesh.hpp"
#include "render/MatrixUtils.hpp"

#include "opencv2/core/core.hpp"
#include "boost/optional/optional.hpp"
#ifdef WITH_RENDER_QOPENGL
	#include <QMatrix4x4>
#endif

namespace render {

/**
 * Desc
 * Coordinate systems:
 * When specifying the vertices: +x = right, +y = up, we look into -z.
 * So z = 0.5 is in front of 0.0.
 * Z-Buffer: 
 * 
 */
class SoftwareRenderer
{
public:
	//SoftwareRenderer();
	SoftwareRenderer(unsigned int viewportWidth, unsigned int viewportHeight);

	bool doBackfaceCulling = false; ///< If true, only draw triangles with vertices ordered CCW in screen-space
	bool doTexturing = false; ///< Desc.

#ifdef WITH_RENDER_QOPENGL
	std::pair<cv::Mat, cv::Mat> render(Mesh mesh, QMatrix4x4 mvp);
#endif
	std::pair<cv::Mat, cv::Mat> render(Mesh mesh, cv::Mat mvp);

	cv::Vec3f projectVertex(cv::Vec4f vertex, cv::Mat mvp);
	
	void enableTexturing(bool doTexturing) {
		this->doTexturing = doTexturing;
	};
	
	void setCurrentTexture(std::shared_ptr<Texture> texture) {
		currentTexture = texture;
	};

private:
	cv::Mat colorBuffer;
	cv::Mat depthBuffer;
	unsigned int viewportWidth = 640;
	unsigned int viewportHeight = 480;
	float aspect;

	// Texturing:
	std::shared_ptr<Texture> currentTexture;
	float dudx, dudy, dvdx, dvdy; // partial derivatives of U/V coordinates with respect to X/Y pixel's screen coordinates

	boost::optional<TriangleToRasterize> processProspectiveTri(Vertex v0, Vertex v1, Vertex v2);

	void rasterTriangle(TriangleToRasterize triangle);

	double implicitLine(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2);

	std::vector<Vertex> clipPolygonToPlaneIn4D(const std::vector<Vertex>& vertices, const cv::Vec4f& planeNormal);

	bool areVerticesCCWInScreenSpace(const Vertex& v0, const Vertex& v1, const Vertex& v2);

	cv::Vec3f tex2D(const cv::Vec2f& texCoord);

	cv::Vec3f tex2D_linear_mipmap_linear(const cv::Vec2f& texCoord);

	cv::Vec2f texCoord_wrap(const cv::Vec2f& texCoord);

	cv::Vec3f tex2D_linear(const cv::Vec2f& imageTexCoord, unsigned char mipmapIndex);

	float clamp(float x, float a, float b);
};

 } /* namespace render */

#endif /* SOFTWARERENDERER_HPP_ */
