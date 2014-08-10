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
 * ...ortho etc:
 * 
 * Shirley: Specify n and f with negative values. which makes sense b/c the points
 * are along the -z axis.
 * Consequences: notably: orthogonal(2, 3): Shirley has denominator (n-f). 
 * In what space are the points in Shirley after this?
 * OGL: We're in the orthographic viewing volume looking down -z.
 * However, n and f are specified positive.

 * B/c the 3D points in front of the cam obviously still have negative z values, the
 * z-value is negated. So: n = 0.1, f = 100; With the given OpenGL ortho matrix,
 * it means a point on the near-plane which will have z = -0.1 will land up
 * on z_clip (which equals z_ndc with ortho because w=1) = -1, and a point on
 * the far plane z = -100 will have z_ndc = +1.
 *
 * That's also why in the perspective case, w_clip is set to -z_eye because
 * to project a point the formula is $x_p = (-n * x_e)/z_e$ (because our near is
 * specified with positive values, but the near-plane is _really_ at -n); but now we
 * just move the minus-sign to the denominator, $x_p = (n * x_e)/-z_e$, so in the projection matrix we can use
 * the (positive) n and f values and afterwards we divide by w = -z_e.
 * 
 * http://www.songho.ca/opengl/gl_projectionmatrix.html
 *
 * Random notes:
 * clip-space: after applying the projection matrix.
 * ndc: after division by w
 * NDC cube: the range of x-coordinate from [l, r] to [-1, 1], the y-coordinate from [b, t] to [-1, 1] and the z-coordinate from [n, f] to [-1, 1].
 *
 * Note/Todo: I read that in screen space, OpenGL transform the z-values again to be between 0 and 1?
 *
 * Similar to OGL, this renderer has a state.
 * Before each render() call, clearBuffers should be called if desired.
 *
 *
 * Here's the whole pipeline:
 * Model space
 * -> model transforms
 * World space
 * -> camera (view/eye) transform
 * View / eye / camera space ("truncated pyramid frustum". In case of ortho, it's already rectangular.)
 * -> perspective/ortho projection
 * Clip coords (x_c, y_c, z_c, w_c); z is [-1, 1] in case of ortho, but not yet in case of persp. But this is fine as we test against w_c?
 * do frustum culling (clipping) here. Test the clip-coords with w_c, and discard.
 * "Then, OpenGL will reconstruct the edges of the polygon where clipping occurs."
 * -> Then divide by the w component of the clip coordinates
 * NDC. (now only 3D vectors: [x_ndc, y_ndc, z_ndc])
 * -> window transform. (also, OGL does some more to the z-buffer?)
 * Screen / window space
 *
 * Add: When do we do clipping & stuff?
 * "both clipping (frustum culling) and NDC transformations are integrated into GL_PROJECTION matrix"
 *
 * Note: In both the ortho and persp case, points at z=-n end up at -1, z=-f at +1. In case of persp proj., this happens only after the divide by w.
 */
class SoftwareRenderer
{
public:
	//SoftwareRenderer();
	SoftwareRenderer(unsigned int viewportWidth, unsigned int viewportHeight);

	bool doBackfaceCulling = false; ///< If true, only draw triangles with vertices ordered CCW in screen-space
	bool doTexturing = false; ///< Desc.

#ifdef WITH_RENDER_QOPENGL
	std::pair<cv::Mat, cv::Mat> render(Mesh mesh, QMatrix4x4 modelViewMatrix, QMatrix4x4 projectionMatrix);
#endif
	// Note: returns a reference (Mat) to the framebuffer, not
	// a clone! I.e. if you don't want your image to get
	// overwritten by a second call to render(...), you have to
	// clone.
	// maybe change and pass depthBuffer as an optional arg (&?), because usually we never need it outside the renderer. Or maybe even a getDepthBuffer().
	// modelViewMatrix goes to eye-space (camera space), projection does ortho or perspective proj.
	std::pair<cv::Mat, cv::Mat> render(Mesh mesh, cv::Mat modelViewMatrix, cv::Mat projectionMatrix);

	// clears the color- and depth buffer
	void clearBuffers();

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

	// Todo: Split this function into the general (core-part) and the texturing part.
	// Then, utils::extractTexture can re-use the core-part.
	boost::optional<TriangleToRasterize> processProspectiveTri(Vertex v0, Vertex v1, Vertex v2);

	void rasterTriangle(TriangleToRasterize triangle);

	std::vector<Vertex> clipPolygonToPlaneIn4D(const std::vector<Vertex>& vertices, const cv::Vec4f& planeNormal);

	cv::Vec3f tex2D(const cv::Vec2f& texCoord);

	cv::Vec3f tex2D_linear_mipmap_linear(const cv::Vec2f& texCoord);

	cv::Vec2f texCoord_wrap(const cv::Vec2f& texCoord);

	cv::Vec3f tex2D_linear(const cv::Vec2f& imageTexCoord, unsigned char mipmapIndex);

	float clamp(float x, float a, float b); // Todo: Document! x, a, b?
};

 } /* namespace render */

#endif /* SOFTWARERENDERER_HPP_ */
