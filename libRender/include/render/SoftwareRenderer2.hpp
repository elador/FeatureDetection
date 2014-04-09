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
#include "boost/optional/optional.hpp"
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
	std::pair<cv::Mat, cv::Mat> render(Mesh mesh, QMatrix4x4 mvp) {
		cv::Mat ocv_mvp = (cv::Mat_<float>(4, 4) <<
			mvp(0, 0), mvp(0, 1), mvp(0, 2), mvp(0, 3),
			mvp(1, 0), mvp(1, 1), mvp(1, 2), mvp(1, 3),
			mvp(2, 0), mvp(2, 1), mvp(2, 2), mvp(2, 3),
			mvp(3, 0), mvp(3, 1), mvp(3, 2), mvp(3, 3));
		return render(mesh, ocv_mvp);
	};
	std::pair<cv::Mat, cv::Mat> render(Mesh mesh, Mat mvp) {
		this->colorBuffer = Mat::zeros(viewportHeight, viewportWidth, CV_8UC4);
		depthBuffer = Mat::ones(viewportHeight, viewportWidth, CV_64FC1) * 1000000;
		std::vector<TriangleToRasterize> trisToRaster;

		// Actual Vertex shader:
		//processedVertex = shade(Vertex); // processedVertex : pos, col, tex, texweight
		std::vector<Vertex> clipSpaceVertices;
		for (const auto& v : mesh.vertex) {
			Mat mpnew = mvp * Mat(v.position);
			clipSpaceVertices.push_back(Vertex(mpnew, v.color, v.texcrd));
		}

		// We're in NDC now (= clip space, clipping volume)
		// PREPARE rasterizer:
		// processProspectiveTriangleToRasterize:
		// for every vertex/tri:
		for (const auto& triIndices : mesh.tvi) {
			// classify vertices visibility with respect to the planes of the view frustum
			// all vertices are not visible - reject the triangle.
			// all vertices are visible - pass the whole triangle to the rasterizer.
			boost::optional<TriangleToRasterize> t = processProspectiveTri(clipSpaceVertices[triIndices[0]], clipSpaceVertices[triIndices[1]], clipSpaceVertices[triIndices[2]]); // true/false?
			if (t) {
				trisToRaster.push_back(*t);
			}
			// at this moment the triangle is known to be intersecting one of the view frustum's planes
			// split the tri etc... then pass to to the rasterizer.
		}

		// runPixelProcessor:
		// Fragment shader: Color the pixel values
		// for every tri:
		for (const auto& tri : trisToRaster) {
			rasterTriangle(tri);
		}
		return std::make_pair(colorBuffer, depthBuffer);
	};

private:
	Mat colorBuffer;
	Mat depthBuffer;
	unsigned int viewportWidth = 640;
	unsigned int viewportHeight = 480;
	float aspect;

	boost::optional<TriangleToRasterize> processProspectiveTri(Vertex v0, Vertex v1, Vertex v2) {
		TriangleToRasterize t;
		t.v0 = v0;	// no memcopy I think. the transformed vertices don't get copied and exist only once. They are a local variable in runVertexProcessor(), the ref is passed here, and if we need to rasterize it, it gets push_back'ed (=copied?) to trianglesToRasterize. Perfect I think. TODO: Not anymore, no ref here
		t.v1 = v1;
		t.v2 = v2;

		// calc 1/v0.w, 1/v1.w, 1/v2.w. No, only for perspective projection

		// divide by w
		// if ortho, we can do the divide as well, it will just be a / 1.0f.
		t.v0.position = t.v0.position / t.v0.position[3];
		t.v1.position = t.v1.position / t.v1.position[3];
		t.v2.position = t.v2.position / t.v2.position[3];

		// project from 4D to 2D window position with depth value in z coordinate
		// Viewport transform:
		//float x_w = (res.x() + 1)*(viewportWidth / 2.0f) + 0.0f; // OpenGL viewport transform (from NDC to viewport) (NDC=clipspace?)
		//float y_w = (res.y() + 1)*(viewportHeight / 2.0f) + 0.0f;
		//y_w = viewportHeight - y_w; // Qt: Origin top-left. OpenGL: bottom-left. OCV: top-left.
		t.v0.position[0] = (t.v0.position[0] + 1) * (viewportWidth / 2.0f);
		t.v0.position[1] = (t.v0.position[1] + 1) * (viewportHeight / 2.0f);
		t.v0.position[1] = viewportHeight - t.v0.position[1];
		t.v1.position[0] = (t.v1.position[0] + 1) * (viewportWidth / 2.0f);
		t.v1.position[1] = (t.v1.position[1] + 1) * (viewportHeight / 2.0f);
		t.v1.position[1] = viewportHeight - t.v1.position[1];
		t.v2.position[0] = (t.v2.position[0] + 1) * (viewportWidth / 2.0f);
		t.v2.position[1] = (t.v2.position[1] + 1) * (viewportHeight / 2.0f);
		t.v2.position[1] = viewportHeight - t.v2.position[1];
		Mat img = Mat::zeros(viewportHeight, viewportWidth, CV_8UC4);
		cv::line(img, cv::Point(t.v0.position[0], t.v0.position[1]), cv::Point(t.v1.position[0], t.v1.position[1]), cv::Scalar(255.0f, 0.0f, 0.0f));
		cv::line(img, cv::Point(t.v1.position[0], t.v1.position[1]), cv::Point(t.v2.position[0], t.v2.position[1]), cv::Scalar(255.0f, 0.0f, 0.0f));
		cv::line(img, cv::Point(t.v2.position[0], t.v2.position[1]), cv::Point(t.v0.position[0], t.v0.position[1]), cv::Scalar(255.0f, 0.0f, 0.0f));
		// My last SW-renderer was:
		// x_w = (x *  vW/2) + vW/2; // equivalent to above
		// y_w = (y * -vH/2) + vH/2; // equiv? Todo!
		// CG book says: (check!)
		// x_w = (x *  vW/2) + (vW-1)/2;
		// y_w = (y * -vH/2) + (vH-1)/2;

		//if (!areVerticesCCWInScreenSpace(t.v0, t.v1, t.v2)) return;

		// find bounding box for the triangle
		/*t.minX = std::max(std::min(t.v0.position[0], std::min(t.v1.position[0], t.v2.position[0])), 0.0f);
		t.maxX = std::min(std::max(t.v0.position[0], std::max(t.v1.position[0], t.v2.position[0])), (float)(viewportWidth - 1));
		t.minY = std::max(std::min(t.v0.position[1], std::min(t.v1.position[1], t.v2.position[1])), 0.0f);
		t.maxY = std::min(std::max(t.v0.position[1], std::max(t.v1.position[1], t.v2.position[1])), (float)(viewportHeight - 1));*/
		t.minX = std::max(std::min(std::floor(t.v0.position[0]), std::min(std::floor(t.v1.position[0]), std::floor(t.v2.position[0]))), 0.0f);
		t.maxX = std::min(std::max(std::ceil(t.v0.position[0]), std::max(std::ceil(t.v1.position[0]), std::ceil(t.v2.position[0]))), (float)(viewportWidth - 1));
		t.minY = std::max(std::min(std::floor(t.v0.position[1]), std::min(std::floor(t.v1.position[1]), std::floor(t.v2.position[1]))), 0.0f);
		t.maxY = std::min(std::max(std::ceil(t.v0.position[1]), std::max(std::ceil(t.v1.position[1]), std::ceil(t.v2.position[1]))), (float)(viewportHeight - 1));

		if (t.maxX <= t.minX || t.maxY <= t.minY)
			return boost::none;

		// Use the triangle:
		
		// barycentric blabla, partial derivatives??? ... (what is for texturing, what for persp., what for rest?)
		
		// Use t
		return boost::optional<TriangleToRasterize>(t);
	};

	void rasterTriangle(TriangleToRasterize triangle) {
		TriangleToRasterize t = triangle; // delete
		for (int yi = t.minY; yi <= t.maxY; yi++)
		{
			for (int xi = t.minX; xi <= t.maxX; xi++)
			{
				// we want centers of pixels to be used in computations. TODO: Do we?
				float x = (float)xi + 0.5f;
				float y = (float)yi + 0.5f;

				// these will be used for barycentric weights computation
				t.one_over_v0ToLine12 = 1.0 / implicitLine(t.v0.position[0], t.v0.position[1], t.v1.position, t.v2.position);
				t.one_over_v1ToLine20 = 1.0 / implicitLine(t.v1.position[0], t.v1.position[1], t.v2.position, t.v0.position);
				t.one_over_v2ToLine01 = 1.0 / implicitLine(t.v2.position[0], t.v2.position[1], t.v0.position, t.v1.position);
				// affine barycentric weights
				double alpha = implicitLine(x, y, t.v1.position, t.v2.position) * t.one_over_v0ToLine12;
				double beta = implicitLine(x, y, t.v2.position, t.v0.position) * t.one_over_v1ToLine20;
				double gamma = implicitLine(x, y, t.v0.position, t.v1.position) * t.one_over_v2ToLine01;

				// if pixel (x, y) is inside the triangle or on one of its edges
				if (alpha >= 0 && beta >= 0 && gamma >= 0)
				{
					//int pixelIndex = (screenHeight - 1 - yi)*screenWidth + xi;
					//int pixelIndexRow = (screenHeight - 1 - yi);
					int pixelIndexRow = yi;
					int pixelIndexCol = xi;

					double z_affine = alpha*(double)t.v0.position[2] + beta*(double)t.v1.position[2] + gamma*(double)t.v2.position[2];	// z

					if (z_affine < depthBuffer.at<double>(pixelIndexRow, pixelIndexCol) && z_affine <= 1.0)
					{
						// attributes interpolation
						cv::Vec3f color_persp = alpha*t.v0.color + beta*t.v1.color + gamma*t.v2.color;
						cv::Vec3f pixelColor = color_persp;
						// clamp bytes to 255
						unsigned char red = (unsigned char)(255.0f * std::min(pixelColor[0], 1.0f));
						unsigned char green = (unsigned char)(255.0f * std::min(pixelColor[1], 1.0f));
						unsigned char blue = (unsigned char)(255.0f * std::min(pixelColor[2], 1.0f));

						// update buffers
						colorBuffer.at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[0] = blue;
						colorBuffer.at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[1] = green;
						colorBuffer.at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[2] = red;
						colorBuffer.at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[3] = 255; // alpha, or 1.0f?
						depthBuffer.at<double>(pixelIndexRow, pixelIndexCol) = z_affine;
					}
				}
			}
		}
	};

	double implicitLine(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2) {
		return ((double)v1[1] - (double)v2[1])*(double)x + ((double)v2[0] - (double)v1[0])*(double)y + (double)v1[0] * (double)v2[1] - (double)v2[0] * (double)v1[1];
	};
	
};

 } /* namespace render */

#endif /* SOFTWARERENDERER2_HPP_ */
