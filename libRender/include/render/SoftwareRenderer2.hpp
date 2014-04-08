/*
 * SoftwareRenderer2.hpp
 *
 *  Created on: 25.11.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef SOFTWARERENDERER2_HPP_
#define SOFTWARERENDERER2_HPP_

#include "render/Mesh.hpp"
#include "render/MatrixUtils.hpp"

#include "opencv2/core/core.hpp"
#include <QMatrix4x4>

using cv::Mat;
using cv::Vec4f;
using std::vector;

namespace render {

/**
 * Desc
 */
class SoftwareRenderer2
{
public:
	//SoftwareRenderer2();
	//SoftwareRenderer2(unsigned int screenWidth, unsigned int screenHeight);
	//~SoftwareRenderer2();

	//ifdef WITH_RENDER_QT? WITH_RENDER_QOPENGL?
	void render(Mesh mesh, QMatrix4x4 mvp) {
		cv::Mat ocv_mvp = (cv::Mat_<float>(4, 4) <<
			mvp(0, 0), mvp(0, 1), mvp(0, 2), mvp(0, 3),
			mvp(1, 0), mvp(1, 1), mvp(1, 2), mvp(1, 3),
			mvp(2, 0), mvp(2, 1), mvp(2, 2), mvp(2, 3),
			mvp(3, 0), mvp(3, 1), mvp(3, 2), mvp(3, 3));
		render(mesh, ocv_mvp);
	};
	void render(Mesh mesh, Mat mvp) {
		std::vector<TriangleToRasterize> trisToRaster;

		// Actual Vertex shader:
		//processedVertex = shade(Vertex); // processedVertex : pos, col, tex, texweight
		std::vector<Vertex> processedVertices;
		for (const auto& v : mesh.vertex) {
			Mat mpnew = mvp * Mat(v.position);
			processedVertices.push_back(Vertex(mpnew, v.color, v.texcrd));
		}

		// We're in NDC now (= clip space, clipping volume)
		// for every vertex/tri:
		for (const auto& triIndices : mesh.tvi) {
			// classify vertices visibility with respect to the planes of the view frustum
			// all vertices are not visible - reject the triangle.
			// all vertices are visible - pass the whole triangle to the rasterizer.
			TriangleToRasterize tri;
			tri.v0 = processedVertices[triIndices[0]];
			tri.v1 = processedVertices[triIndices[1]];
			tri.v2 = processedVertices[triIndices[2]];
			trisToRaster.push_back(tri); // better: call prepare(v0, v1, v2) and check the tri. And below, push_back.
			// at this moment the triangle is known to be intersecting one of the view frustum's planes
			// split the tri etc... then pass to to the rasterizer.
		}
		// PREPARE rasterizer:
		// processProspectiveTriangleToRasterize:
		// for every tri:
		for (const auto& tri : trisToRaster) {
			// calc 1/v0.w, 1/v1.w, 1/v2.w
			// divide by w
			// if ortho, we can do the divide as well, it will just be a / 1.0f.

			// Viewport Transform
			// if (!areVerticesCCWInScreenSpace(t.v0, t.v1, t.v2)), return;
			// find bounding box for the triangle
			// barycentric blabla, partial derivatives??? ... (what is for texturing, what for persp., what for rest?)
		}
		// Viewport transform:
		//float x_w = (res.x() + 1)*(viewportWidth / 2.0f) + 0.0f; // OpenGL viewport transform (from NDC to viewport) (NDC=clipspace?)
		//float y_w = (res.y() + 1)*(viewportHeight / 2.0f) + 0.0f;
		//y_w = viewportHeight - y_w; // Qt: Origin top-left. OpenGL: bottom-left. OCV: top-left.
		// My last SW-renderer was:
		// x_w = (x *  vW/2) + vW/2; // equivalent to above
		// y_w = (y * -vH/2) + vH/2; // equiv? Todo!
		// CG book says: (check!)
		// x_w = (x *  vW/2) + (vW-1)/2;
		// y_w = (y * -vH/2) + (vH-1)/2;

		// runPixelProcessor:
		// Fragment shader: Color the pixel values
		// for every tri:
		// loop over min/max
		// calc bary
		// if visible according to z-buffer:
		// interpolate tex, color
		// color the pixel: runPixelShader: just returns color or return tex2D(texture, texCoords)
		// clamp
		// set pixel value in framebuffer, set depth-buffer
	};

private:
	Mat colorBuffer;
	Mat depthBuffer;
	unsigned int screenWidth;
	unsigned int screenHeight;
	float aspect;
	
};

 } /* namespace render */

#endif /* SOFTWARERENDERER2_HPP_ */
