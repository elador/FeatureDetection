/*
 * MeshUtils.cpp
 *
 *  Created on: 12.12.2012
 *      Author: Patrik Huber
 */

#include "render/MeshUtils.hpp"
#include "render/utils.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <array>
#include <iostream>
#include <fstream>

using cv::Mat;
using cv::Point2f;
using cv::Vec2f;
using cv::Scalar;

namespace render {
	namespace utils {

Mesh MeshUtils::createCube()
{
	Mesh cube;
	cube.vertex.resize(24);

	for (int i = 0; i < 24; i++)
		cube.vertex[i].color = cv::Vec3f(1.0f, 0.0f, 0.0f);

	cube.vertex[0].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[0].texcrd = Vec2f(0.0f, 0.0f);
	cube.vertex[1].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[1].texcrd = Vec2f(0.0f, 1.0f);
	cube.vertex[2].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[2].texcrd = Vec2f(1.0f, 1.0f);
	cube.vertex[3].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[3].texcrd = Vec2f(1.0f, 0.0f);

	cube.vertex[4].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[4].texcrd = Vec2f(0.0f, 0.0f);
	cube.vertex[5].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[5].texcrd = Vec2f(0.0f, 1.0f);
	cube.vertex[6].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[6].texcrd = Vec2f(1.0f, 1.0f);
	cube.vertex[7].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[7].texcrd = Vec2f(1.0f, 0.0f);

	cube.vertex[8].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[8].texcrd = Vec2f(0.0f, 0.0f);
	cube.vertex[9].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[9].texcrd = Vec2f(0.0f, 1.0f);
	cube.vertex[10].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[10].texcrd = Vec2f(1.0f, 1.0f);
	cube.vertex[11].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[11].texcrd = Vec2f(1.0f, 0.0f);

	cube.vertex[12].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[12].texcrd = Vec2f(0.0f, 0.0f);
	cube.vertex[13].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[13].texcrd = Vec2f(0.0f, 1.0f);
	cube.vertex[14].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[14].texcrd = Vec2f(1.0f, 1.0f);
	cube.vertex[15].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[15].texcrd = Vec2f(1.0f, 0.0f);

	cube.vertex[16].position = cv::Vec4f(-0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[16].texcrd = Vec2f(0.0f, 0.0f);
	cube.vertex[17].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[17].texcrd = Vec2f(0.0f, 1.0f);
	cube.vertex[18].position = cv::Vec4f(0.5f, 0.5f, 0.5f, 1.0f);
	cube.vertex[18].texcrd = Vec2f(1.0f, 1.0f);
	cube.vertex[19].position = cv::Vec4f(0.5f, 0.5f, -0.5f, 1.0f);
	cube.vertex[19].texcrd = Vec2f(1.0f, 0.0f);

	cube.vertex[20].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[20].texcrd = Vec2f(0.0f, 0.0f);
	cube.vertex[21].position = cv::Vec4f(-0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[21].texcrd = Vec2f(0.0f, 1.0f);
	cube.vertex[22].position = cv::Vec4f(0.5f, -0.5f, -0.5f, 1.0f);
	cube.vertex[22].texcrd = Vec2f(1.0f, 1.0f);
	cube.vertex[23].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	cube.vertex[23].texcrd = Vec2f(1.0f, 0.0f);

	// the efficiency of this might be improvable...
	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 1; vi[2] = 2;
	cube.tvi.push_back(vi);
	vi[0] = 0; vi[1] = 2; vi[2] = 3;
	cube.tvi.push_back(vi);
	vi[0] = 4; vi[1] = 5; vi[2] = 6;
	cube.tvi.push_back(vi);
	vi[0] = 4; vi[1] = 6; vi[2] = 7;
	cube.tvi.push_back(vi);
	vi[0] = 8; vi[1] = 9; vi[2] = 10;
	cube.tvi.push_back(vi);
	vi[0] = 8; vi[1] = 10; vi[2] = 11;
	cube.tvi.push_back(vi);
	vi[0] = 12; vi[1] = 13; vi[2] = 14;
	cube.tvi.push_back(vi);
	vi[0] = 12; vi[1] = 14; vi[2] = 15;
	cube.tvi.push_back(vi);
	vi[0] = 16; vi[1] = 17; vi[2] = 18;
	cube.tvi.push_back(vi);
	vi[0] = 16; vi[1] = 18; vi[2] = 19;
	cube.tvi.push_back(vi);
	vi[0] = 20; vi[1] = 21; vi[2] = 22;
	cube.tvi.push_back(vi);
	vi[0] = 20; vi[1] = 22; vi[2] = 23;
	cube.tvi.push_back(vi);
	/*cube.triangleList.push_back(render::Triangle(cube.vertex[0], cube.vertex[1], cube.vertex[2]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[0], cube.vertex[2], cube.vertex[3]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[4], cube.vertex[5], cube.vertex[6]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[4], cube.vertex[6], cube.vertex[7]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[8], cube.vertex[9], cube.vertex[10]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[8], cube.vertex[10], cube.vertex[11]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[12], cube.vertex[13], cube.vertex[14]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[12], cube.vertex[14], cube.vertex[15]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[16], cube.vertex[17], cube.vertex[18]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[16], cube.vertex[18], cube.vertex[19]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[20], cube.vertex[21], cube.vertex[22]));
	cube.triangleList.push_back(render::Triangle(cube.vertex[20], cube.vertex[22], cube.vertex[23]));*/

	cube.tci = cube.tvi;

	cube.texture = std::make_shared<Texture>();
	cube.texture->createFromFile("C:\\Users\\Patrik\\Documents\\Github\\img.png");
	cube.hasTexture = true;

	return cube;
}

Mesh MeshUtils::createPlane()
{
	Mesh plane;
	plane.vertex.resize(4);
	plane.vertex[0].position = cv::Vec4f(-0.5f, 0.0f, -0.5f, 1.0f);
	plane.vertex[1].position = cv::Vec4f(-0.5f, 0.0f, 0.5f, 1.0f);
	plane.vertex[2].position = cv::Vec4f(0.5f, 0.0f, 0.5f, 1.0f);
	plane.vertex[3].position = cv::Vec4f(0.5f, 0.0f, -0.5f, 1.0f);
	/*
	plane.vertex[0].position = cv::Vec4f(-1.0f, 1.0f, 0.0f);
	plane.vertex[1].position = cv::Vec4f(-1.0f, -1.0f, 0.0f);
	plane.vertex[2].position = cv::Vec4f(1.0f, -1.0f, 0.0f);
	plane.vertex[3].position = cv::Vec4f(1.0f, 1.0f, 0.0f);
	*/
	plane.vertex[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);
	plane.vertex[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);
	plane.vertex[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	plane.vertex[3].color = cv::Vec3f(1.0f, 1.0f, 1.0f);

	plane.vertex[0].texcrd = Vec2f(0.0f, 0.0f);
	plane.vertex[1].texcrd = Vec2f(0.0f, 4.0f);
	plane.vertex[2].texcrd = Vec2f(4.0f, 4.0f);
	plane.vertex[3].texcrd = Vec2f(4.0f, 0.0f);

	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 1; vi[2] = 2;
	plane.tvi.push_back(vi);
	vi[0] = 0; vi[1] = 2; vi[2] = 3;
	plane.tvi.push_back(vi);

	//plane.triangleList.push_back(render::Triangle(plane.vertex[0], plane.vertex[1], plane.vertex[2]));
	//plane.triangleList.push_back(render::Triangle(plane.vertex[0], plane.vertex[2], plane.vertex[3]));

	plane.texture = std::make_shared<Texture>();
	plane.texture->createFromFile("C:\\Users\\Patrik\\Cloud\\PhD\\rocks.png");
	plane.hasTexture = true;

	return plane;
}

Mesh MeshUtils::createPyramid()
{
	Mesh pyramid;
	pyramid.vertex.resize(4);

	pyramid.vertex[0].position = cv::Vec4f(-0.5f, 0.0f, 0.5f, 1.0f);
	pyramid.vertex[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);

	pyramid.vertex[1].position = cv::Vec4f(0.5f, 0.0f, 0.5f, 1.0f);
	pyramid.vertex[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);

	pyramid.vertex[2].position = cv::Vec4f(0.0f, 0.0f, -0.5f, 1.0f);
	pyramid.vertex[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	
	pyramid.vertex[3].position = cv::Vec4f(0.0f, 1.0f, 0.0f, 1.0f);
	pyramid.vertex[3].color = cv::Vec3f(0.5f, 0.5f, 0.5f);

	// the efficiency of this might be improvable...
	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 2; vi[2] = 1; // the bottom plate, draw so that visible from below
	pyramid.tvi.push_back(vi);
	vi[0] = 0; vi[1] = 1; vi[2] = 3; // front
	pyramid.tvi.push_back(vi);
	vi[0] = 2; vi[1] = 3; vi[2] = 1; // right side
	pyramid.tvi.push_back(vi);
	vi[0] = 3; vi[1] = 2; vi[2] = 0; // left side
	pyramid.tvi.push_back(vi);

	pyramid.hasTexture = false;

	return pyramid;
}

shared_ptr<Mesh> MeshUtils::createTriangle()
{
	shared_ptr<Mesh> triangle = std::make_shared<Mesh>();
	triangle->vertex.resize(3);

	triangle->vertex[0].position = cv::Vec4f(-0.5f, 0.5f, 0.5f, 1.0f);
	triangle->vertex[0].color = cv::Vec3f(1.0f, 0.0f, 0.0f);
	triangle->vertex[0].texcrd = Vec2f(0.0f, 0.0f);
	
	triangle->vertex[1].position = cv::Vec4f(-0.5f, -0.5f, 0.5f, 1.0f);
	triangle->vertex[1].color = cv::Vec3f(0.0f, 1.0f, 0.0f);
	triangle->vertex[1].texcrd = Vec2f(0.0f, 1.0f);
	
	triangle->vertex[2].position = cv::Vec4f(0.5f, -0.5f, 0.5f, 1.0f);
	triangle->vertex[2].color = cv::Vec3f(0.0f, 0.0f, 1.0f);
	triangle->vertex[2].texcrd = Vec2f(1.0f, 1.0f);
	

	// the efficiency of this might be improvable...
	std::array<int, 3> vi;
	vi[0] = 0; vi[1] = 1; vi[2] = 2;
	triangle->tvi.push_back(vi);
	
	//triangle->texture = std::make_shared<Texture>();
	//triangle->texture->createFromFile("C:\\Users\\Patrik\\Cloud\\PhD\\up.png");
	triangle->hasTexture = false;

	return triangle;
}

Mat MeshUtils::drawTexCoords(Mesh mesh)
{
	Mat texImg(512, 512, CV_8UC4, cv::Scalar(0.0f, 0.0f, 0.0f, 255.0f));
	for (const auto& triIdx : mesh.tvi) {
		cv::line(texImg, Point2f(mesh.vertex[triIdx[0]].texcrd[0] * texImg.cols, mesh.vertex[triIdx[0]].texcrd[1] * texImg.rows), Point2f(mesh.vertex[triIdx[1]].texcrd[0] * texImg.cols, mesh.vertex[triIdx[1]].texcrd[1] * texImg.rows), Scalar(255.0f, 0.0f, 0.0f));
		cv::line(texImg, Point2f(mesh.vertex[triIdx[1]].texcrd[0] * texImg.cols, mesh.vertex[triIdx[1]].texcrd[1] * texImg.rows), Point2f(mesh.vertex[triIdx[2]].texcrd[0] * texImg.cols, mesh.vertex[triIdx[2]].texcrd[1] * texImg.rows), Scalar(255.0f, 0.0f, 0.0f));
		cv::line(texImg, Point2f(mesh.vertex[triIdx[2]].texcrd[0] * texImg.cols, mesh.vertex[triIdx[2]].texcrd[1] * texImg.rows), Point2f(mesh.vertex[triIdx[0]].texcrd[0] * texImg.cols, mesh.vertex[triIdx[0]].texcrd[1] * texImg.rows), Scalar(255.0f, 0.0f, 0.0f));
	}
	return texImg;
}

// Returns true if inside the tri or on the border
bool MeshUtils::isPointInTriangle(cv::Point2f point, cv::Point2f triV0, cv::Point2f triV1, cv::Point2f triV2) {
	/* See http://www.blackpawn.com/texts/pointinpoly/ */
	// Compute vectors        
	cv::Point2f v0 = triV2 - triV0;
	cv::Point2f v1 = triV1 - triV0;
	cv::Point2f v2 = point - triV0;

	// Compute dot products
	float dot00 = v0.dot(v0);
	float dot01 = v0.dot(v1);
	float dot02 = v0.dot(v2);
	float dot11 = v1.dot(v1);
	float dot12 = v1.dot(v2);

	// Compute barycentric coordinates
	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is in triangle
	return (u >= 0) && (v >= 0) && (u + v < 1);
}

// image: where to extract the texture from
// note: framebuffer should have size of the image (ok not necessarily. What about mobile?) (well it should, to get optimal quality (and everywhere the same quality)?)
// note: mvpMatrix: Atm working with a 4x4 (full) affine. But anything would work, just take care with the w-division.
// Regarding the depth-buffer: We could also pass an instance of a Renderer here. Depending on how "stateful" the renderer is, this might make more sense.
Mat extractTexture(Mesh mesh, Mat mvpMatrix, int viewportWidth, int viewportHeight, Mat image, Mat depthBuffer) {
	// optional param Mat textureMap = Mat(512, 512, CV_8UC3) ?
	//Mat textureMap(512, 512, inputImage.type());
	Mat textureMap = Mat::zeros(512, 512, CV_8UC3); // We don't want an alpha channel. We might want to handle grayscale input images though.
	Mat visibilityMask = Mat::zeros(512, 512, CV_8UC3);

	for (const auto& triangleIndices : mesh.tvi) {

		// Find out if the current triangle is visible:
		// We do a second rendering-pass here. We use the depth-buffer of the final image, and then, here,
		// check if each pixel in a triangle is visible. If the whole triangle is visible, we use it to extract
		// the texture.
		// Possible improvement: - If only part of the triangle is visible, split it
		// - Share more code with the renderer?
		Vertex v0_3d = mesh.vertex[triangleIndices[0]];
		Vertex v1_3d = mesh.vertex[triangleIndices[1]];
		Vertex v2_3d = mesh.vertex[triangleIndices[2]];

		Vertex v0, v1, v2; // we don't copy the color and texcoords, we only do the visibility check here.
		// This could be optimized in 2 ways though:
		// - Use render(), or as in render(...), transfer the vertices once, not in a loop over all triangles (vertices are getting transformed multiple times)
		// - We transform them later (below) a second time. Only do it once.
		v0.position = Mat(mvpMatrix * Mat(v0_3d.position));
		v1.position = Mat(mvpMatrix * Mat(v1_3d.position));
		v2.position = Mat(mvpMatrix * Mat(v2_3d.position));

		// Well, in in principle, we'd have to do the whole stuff as in render(), like
		// clipping against the frustums etc.
		// But as long as our model is fully on the screen, we're fine.

		// divide by w
		// if ortho, we can do the divide as well, it will just be a / 1.0f.
		v0.position = v0.position / v0.position[3];
		v1.position = v1.position / v1.position[3];
		v2.position = v2.position / v2.position[3];

		// Todo: This is all very similar to processProspectiveTri(...), except the other function does texturing stuff as well. Remove code duplication!
		Vec2f v0_clip = clipToScreenSpace(Vec2f(v0.position[0], v0.position[1]), viewportWidth, viewportHeight);
		Vec2f v1_clip = clipToScreenSpace(Vec2f(v1.position[0], v1.position[1]), viewportWidth, viewportHeight);
		Vec2f v2_clip = clipToScreenSpace(Vec2f(v2.position[0], v2.position[1]), viewportWidth, viewportHeight);
		v0.position[0] = v0_clip[0]; v0.position[1] = v0_clip[1];
		v1.position[0] = v1_clip[0]; v1.position[1] = v1_clip[1];
		v2.position[0] = v2_clip[0]; v2.position[1] = v2_clip[1];
		
		//if (doBackfaceCulling) {
			if (!areVerticesCCWInScreenSpace(v0, v1, v2))
				continue;
		//}

		cv::Rect bbox = calculateBoundingBox(v0, v1, v2, viewportWidth, viewportHeight);
		int minX = bbox.x;
		int maxX = bbox.x + bbox.width;
		int minY = bbox.y;
		int maxY = bbox.y + bbox.height;

		//if (t.maxX <= t.minX || t.maxY <= t.minY) 	// Note: Can the width/height of the bbox be negative? Maybe we only need to check for equality here?
		//	continue;

		bool wholeTriangleIsVisible = true;
		for (int yi = minY; yi <= maxY; yi++)
		{
			for (int xi = minX; xi <= maxX; xi++)
			{
				// we want centers of pixels to be used in computations. TODO: Do we?
				float x = (float)xi + 0.5f;
				float y = (float)yi + 0.5f;
				// these will be used for barycentric weights computation
				double one_over_v0ToLine12 = 1.0 / utils::implicitLine(v0.position[0], v0.position[1], v1.position, v2.position);
				double one_over_v1ToLine20 = 1.0 / utils::implicitLine(v1.position[0], v1.position[1], v2.position, v0.position);
				double one_over_v2ToLine01 = 1.0 / utils::implicitLine(v2.position[0], v2.position[1], v0.position, v1.position);
				// affine barycentric weights
				double alpha = utils::implicitLine(x, y, v1.position, v2.position) * one_over_v0ToLine12;
				double beta = utils::implicitLine(x, y, v2.position, v0.position) * one_over_v1ToLine20;
				double gamma = utils::implicitLine(x, y, v0.position, v1.position) * one_over_v2ToLine01;
				// if pixel (x, y) is inside the triangle or on one of its edges
				if (alpha >= 0 && beta >= 0 && gamma >= 0)
				{
					double z_affine = alpha*(double)v0.position[2] + beta*(double)v1.position[2] + gamma*(double)v2.position[2];
					// The '<= 1.0' clips against the far-plane in NDC. We clip against the near-plane earlier.
					if (z_affine < depthBuffer.at<double>(yi, xi)/* && z_affine <= 1.0*/) {
						wholeTriangleIsVisible = false;
						break;
					}
					else {
						
					}
				}
			}
			if (!wholeTriangleIsVisible) {
				break;
			}
		}

		if (!wholeTriangleIsVisible) {
			continue;
		}

		cv::Point2f srcTri[3];
		cv::Point2f dstTri[3];
		cv::Vec4f vec(mesh.vertex[triangleIndices[0]].position[0], mesh.vertex[triangleIndices[0]].position[1], mesh.vertex[triangleIndices[0]].position[2], 1.0f);
		cv::Vec4f res = Mat(mvpMatrix * Mat(vec));
		res /= res[3];
		Vec2f screenSpace = utils::clipToScreenSpace(Vec2f(res[0], res[1]), viewportWidth, viewportHeight);
		srcTri[0] = screenSpace;

		vec = cv::Vec4f(mesh.vertex[triangleIndices[1]].position[0], mesh.vertex[triangleIndices[1]].position[1], mesh.vertex[triangleIndices[1]].position[2], 1.0f);
		res = Mat(mvpMatrix * Mat(vec));
		res /= res[3];
		screenSpace = utils::clipToScreenSpace(Vec2f(res[0], res[1]), viewportWidth, viewportHeight);
		srcTri[1] = screenSpace;

		vec = cv::Vec4f(mesh.vertex[triangleIndices[2]].position[0], mesh.vertex[triangleIndices[2]].position[1], mesh.vertex[triangleIndices[2]].position[2], 1.0f);
		res = Mat(mvpMatrix * Mat(vec));
		res /= res[3];
		screenSpace = utils::clipToScreenSpace(Vec2f(res[0], res[1]), viewportWidth, viewportHeight);
		srcTri[2] = screenSpace;

		// ROI in the source image:
		// Todo: Check if the triangle is on screen. If it's outside, we crash here.
		float src_tri_min_x = std::min(srcTri[0].x, std::min(srcTri[1].x, srcTri[2].x)); // note: might be better to round later (i.e. use the float points for getAffineTransform for a more accurate warping)
		float src_tri_max_x = std::max(srcTri[0].x, std::max(srcTri[1].x, srcTri[2].x));
		float src_tri_min_y = std::min(srcTri[0].y, std::min(srcTri[1].y, srcTri[2].y));
		float src_tri_max_y = std::max(srcTri[0].y, std::max(srcTri[1].y, srcTri[2].y));

		Mat inputImageRoi = image.rowRange(cvFloor(src_tri_min_y), cvCeil(src_tri_max_y)).colRange(cvFloor(src_tri_min_x), cvCeil(src_tri_max_x)); // We round down and up. ROI is possibly larger. But wrong pixels get thrown away later when we check if the point is inside the triangle? Correct?
		srcTri[0] -= Point2f(src_tri_min_x, src_tri_min_y);
		srcTri[1] -= Point2f(src_tri_min_x, src_tri_min_y);
		srcTri[2] -= Point2f(src_tri_min_x, src_tri_min_y); // shift all the points to correspond to the roi

		dstTri[0] = cv::Point2f(textureMap.cols*mesh.vertex[triangleIndices[0]].texcrd[0], textureMap.rows*mesh.vertex[triangleIndices[0]].texcrd[1] - 1.0f);
		dstTri[1] = cv::Point2f(textureMap.cols*mesh.vertex[triangleIndices[1]].texcrd[0], textureMap.rows*mesh.vertex[triangleIndices[1]].texcrd[1] - 1.0f);
		dstTri[2] = cv::Point2f(textureMap.cols*mesh.vertex[triangleIndices[2]].texcrd[0], textureMap.rows*mesh.vertex[triangleIndices[2]].texcrd[1] - 1.0f);

		/// Get the Affine Transform
		Mat warp_mat = getAffineTransform(srcTri, dstTri);

		/// Apply the Affine Transform just found to the src image
		Mat tmpDstBuffer = Mat::zeros(textureMap.rows, textureMap.cols, image.type()); // I think using the source-size here is not correct. The dst might be larger. We should warp the endpoints and set to max-w/h. No, I think it would be even better to directly warp to the final textureMap size. (so that the last step is only a 1:1 copy)
		warpAffine(inputImageRoi, tmpDstBuffer, warp_mat, tmpDstBuffer.size(), cv::INTER_CUBIC, cv::BORDER_TRANSPARENT); // last row/col is zeros, depends on interpolation method. Maybe because of rounding or interpolation? So it cuts a little. Maybe try to implement by myself?

		// only copy to final img if point is inside the triangle (or on the border)
		for (int x = std::min(dstTri[0].x, std::min(dstTri[1].x, dstTri[2].x)); x < std::max(dstTri[0].x, std::max(dstTri[1].x, dstTri[2].x)); ++x) {
			for (int y = std::min(dstTri[0].y, std::min(dstTri[1].y, dstTri[2].y)); y < std::max(dstTri[0].y, std::max(dstTri[1].y, dstTri[2].y)); ++y) {
				if (MeshUtils::isPointInTriangle(cv::Point2f(x, y), dstTri[0], dstTri[1], dstTri[2])) {
					textureMap.at<cv::Vec3b>(y, x) = tmpDstBuffer.at<cv::Vec3b>(y, x);
				}
			}
		}
	}
	return textureMap;
}

	} /* namespace utils */
} /* namespace render */
