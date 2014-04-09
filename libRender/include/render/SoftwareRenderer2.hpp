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
 * Coordinate systems:
 * When specifying the vertices: +x = right, +y = up, we look into -z.
 * So z = 0.5 is in front of 0.0.
 * Z-Buffer: 
 * 
 */
class SoftwareRenderer2
{
public:
	//SoftwareRenderer2();
	//SoftwareRenderer2(unsigned int screenWidth, unsigned int screenHeight);
	//~SoftwareRenderer2();

	bool doBackfaceCulling = false; ///< If true, only draw triangles with vertices ordered CCW in screen-space

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
			// Todo: Split this whole stuff up. Make a "clip" function, ... rename "processProspective..".. what is "process"... get rid of "continue;"-stuff by moving stuff inside process...
			// classify vertices visibility with respect to the planes of the view frustum
			// we're in clip-coords (NDC), so just check if outside [-1, 1] x ...
			unsigned char visibilityBits[3];
			for (unsigned char k = 0; k < 3; k++)
			{
				visibilityBits[k] = 0;
				float xOverW = clipSpaceVertices[triIndices[k]].position[0] / clipSpaceVertices[triIndices[k]].position[3];
				float yOverW = clipSpaceVertices[triIndices[k]].position[1] / clipSpaceVertices[triIndices[k]].position[3];
				float zOverW = clipSpaceVertices[triIndices[k]].position[2] / clipSpaceVertices[triIndices[k]].position[3];
				if (xOverW < -1)			// true if outside of view frustum
					visibilityBits[k] |= 1;	// set bit if outside of frustum
				if (xOverW > 1)
					visibilityBits[k] |= 2;
				if (yOverW < -1)
					visibilityBits[k] |= 4;
				if (yOverW > 1)
					visibilityBits[k] |= 8;
				if (zOverW < -1)
					visibilityBits[k] |= 16;
				if (zOverW > 1)
					visibilityBits[k] |= 32;
			} // if all bits are 0, then it's inside the frustum
			// all vertices are not visible - reject the triangle.
			if ((visibilityBits[0] & visibilityBits[1] & visibilityBits[2]) > 0)
			{
				continue;
			}
			// all vertices are visible - pass the whole triangle to the rasterizer. = All bits of all 3 triangles are 0.
			if ((visibilityBits[0] | visibilityBits[1] | visibilityBits[2]) == 0)
			{
				boost::optional<TriangleToRasterize> t = processProspectiveTri(clipSpaceVertices[triIndices[0]], clipSpaceVertices[triIndices[1]], clipSpaceVertices[triIndices[2]]);
				if (t) {
					trisToRaster.push_back(*t);
				}
				continue;
			}
			// at this moment the triangle is known to be intersecting one of the view frustum's planes
			std::vector<Vertex> vertices;
			vertices.push_back(clipSpaceVertices[triIndices[0]]);
			vertices.push_back(clipSpaceVertices[triIndices[1]]);
			vertices.push_back(clipSpaceVertices[triIndices[2]]);
			// split the tri etc... then pass to to the rasterizer.
			//vertices = clipPolygonToPlaneIn4D(vertices, cv::Vec4f(0.0f, 0.0f, -1.0f, -1.0f));	// This is the near-plane, right? Because we only have to check against that. For tlbr planes of the frustum, we can just draw, and then clamp it because it's outside the screen
			vertices = clipPolygonToPlaneIn4D(vertices, cv::Vec4f(0.0f, 0.0f, 1.0f, -1.0f));	// This is the near-plane, right? Because we only have to check against that. For tlbr planes of the frustum, we can just draw, and then clamp it because it's outside the screen
			//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(0.0f, 0.0f, 1.0f, -1.0f));
			//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(-1.0f, 0.0f, 0.0f, -1.0f));
			//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(1.0f, 0.0f, 0.0f, -1.0f));
			//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(0.0f, -1.0f, 0.0f, -1.0f));
			//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(0.0f, 1.0f, 0.0f, -1.0f));
			/* Note from mail: (note: stuff might flip because we change z/P-matrix?)
			vertices = clipPolygonToPlaneIn4D(vertices, cv::Vec4f(0.0f, 0.0f, -1.0f, -1.0f));
			PH: That vector should be the normal of the NEAR-plane of the frustum, right? Because we only have to check if the triangle intersects the near plane. (?) and the rest we should be able to just clamp.
			=> That's right. It's funny here it's actually a 4D hyperplane and it works! Math is beautiful :).
			=> Clipping to the near plane must be done because after w-division tris crossing it would get distorted. Clipping against other planes can be done but I think it's faster to simply check pixel's boundaries during rasterization stage.
			*/

			// triangulation of the polygon formed of vertices array
			if (vertices.size() >= 3)
			{
				for (unsigned char k = 0; k < vertices.size() - 2; k++)
				{
					boost::optional<TriangleToRasterize> t = processProspectiveTri(vertices[0], vertices[1 + k], vertices[2 + k]);
					if (t) {
						trisToRaster.push_back(*t);
					}
				}
			}
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

		if (doBackfaceCulling) {
			if (!areVerticesCCWInScreenSpace(t.v0, t.v1, t.v2))
				return boost::none;
		}

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

	std::vector<Vertex> clipPolygonToPlaneIn4D(const std::vector<Vertex>& vertices, const cv::Vec4f& planeNormal)
	{
		std::vector<Vertex> clippedVertices;

		// We can have 2 cases:
		//	* 1 vertex visible: we make 1 new triangle out of the visible vertex plus the 2 intersection points with the near-plane
		//  * 2 vertices visible: we have a quad, so we have to make 2 new triangles out of it.

		for (unsigned int i = 0; i < vertices.size(); i++)
		{
			int a = i;
			int b = (i + 1) % vertices.size();

			float fa = vertices[a].position.dot(planeNormal);
			float fb = vertices[b].position.dot(planeNormal);

			if ((fa < 0 && fb > 0) || (fa > 0 && fb < 0))
			{
				cv::Vec4f direction = vertices[b].position - vertices[a].position;
				float t = -(planeNormal.dot(vertices[a].position)) / (planeNormal.dot(direction));

				cv::Vec4f position = vertices[a].position + t*direction;
				cv::Vec3f color = vertices[a].color + t*(vertices[b].color - vertices[a].color);
				cv::Vec2f texCoord = vertices[a].texcrd + t*(vertices[b].texcrd - vertices[a].texcrd);	// We could omit that if we don't render with texture.

				if (fa < 0)
				{
					clippedVertices.push_back(vertices[a]);
					clippedVertices.push_back(Vertex(position, color, texCoord));
				}
				else if (fb < 0)
				{
					clippedVertices.push_back(Vertex(position, color, texCoord));
				}
			}
			else if (fa < 0 && fb < 0)
			{
				clippedVertices.push_back(vertices[a]);
			}
		}

		return clippedVertices;
	};

	bool areVerticesCCWInScreenSpace(const Vertex& v0, const Vertex& v1, const Vertex& v2)
	{
		float dx01 = v1.position[0] - v0.position[0];
		float dy01 = v1.position[1] - v0.position[1];
		float dx02 = v2.position[0] - v0.position[0];
		float dy02 = v2.position[1] - v0.position[1];

		return (dx01*dy02 - dy01*dx02 < 0.0f); // Original: (dx01*dy02 - dy01*dx02 > 0.0f). But: OpenCV has origin top-left, y goes down
	};
	
};

 } /* namespace render */

#endif /* SOFTWARERENDERER2_HPP_ */
