/*
* SoftwareRenderer.hpp
*
*  Created on: 06.04.2014
*      Author: Patrik Huber
*/

#include "render/SoftwareRenderer.hpp"

#include "render/utils.hpp"

using cv::Mat;
using cv::Vec4b;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::pair;
using std::make_pair;
using std::vector;
using std::min;
using std::max;
using std::floor;
using std::ceil;

namespace render {

SoftwareRenderer::SoftwareRenderer(unsigned int viewportWidth, unsigned int viewportHeight) : viewportWidth(viewportWidth), viewportHeight(viewportHeight)
{
}

#ifdef WITH_RENDER_QOPENGL
pair<Mat, Mat> SoftwareRenderer::render(Mesh mesh, QMatrix4x4 mvp)
{
	// We assign the values one-by-one since if we used
	// mvp.data() or something, we'd have to transpose
	// because Qt stores the matrix col-major in memory.
	Mat ocv_mvp = (cv::Mat_<float>(4, 4) <<
		mvp(0, 0), mvp(0, 1), mvp(0, 2), mvp(0, 3),
		mvp(1, 0), mvp(1, 1), mvp(1, 2), mvp(1, 3),
		mvp(2, 0), mvp(2, 1), mvp(2, 2), mvp(2, 3),
		mvp(3, 0), mvp(3, 1), mvp(3, 2), mvp(3, 3));
	return render(mesh, ocv_mvp);
}
#endif

pair<Mat, Mat> SoftwareRenderer::render(Mesh mesh, Mat mvp)
{
	colorBuffer = Mat::zeros(viewportHeight, viewportWidth, CV_8UC4);
	depthBuffer = Mat::ones(viewportHeight, viewportWidth, CV_64FC1) * 1000000;
	//depthBuffer = Mat::ones(viewportHeight, viewportWidth, CV_64FC1) * -0.88;

	vector<TriangleToRasterize> trisToRaster;

	// Vertex shader:
	//processedVertex = shade(Vertex); // processedVertex : pos, col, tex, texweight
	vector<Vertex> clipSpaceVertices;
	for (const auto& v : mesh.vertex) {
		Mat mpnew = mvp * Mat(v.position);
		clipSpaceVertices.push_back(Vertex(mpnew, v.color, v.texcrd));
	}

	// We're in clip-space now
	// PREPARE rasterizer:
	// processProspectiveTriangleToRasterize:
	// for every vertex/tri:
	for (const auto& triIndices : mesh.tvi) {
		// Todo: Split this whole stuff up. Make a "clip" function, ... rename "processProspective..".. what is "process"... get rid of "continue;"-stuff by moving stuff inside process...
		// classify vertices visibility with respect to the planes of the view frustum
		// we're in clip-coords (NDC), so just check if outside [-1, 1] x ...
		// Actually we're in clip-coords and it's not the same as NDC. We're only in NDC after the division by w.
		// We should do the clipping in clip-coords though. See http://www.songho.ca/opengl/gl_projectionmatrix.html for more details.
		// However, when comparing against w_c below, we might run into the trouble of the sign again in the affine case.
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
			//if (zOverW < -1) // these 4 lines somehow only clip against the back-plane of the frustum?
				//visibilityBits[k] |= 16;
			//if (zOverW > 1)
				//visibilityBits[k] |= 32;
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
		vector<Vertex> vertices;
		vertices.push_back(clipSpaceVertices[triIndices[0]]);
		vertices.push_back(clipSpaceVertices[triIndices[1]]);
		vertices.push_back(clipSpaceVertices[triIndices[2]]);
		// split the tri etc... then pass to to the rasterizer.
		//vertices = clipPolygonToPlaneIn4D(vertices, Vec4f(0.0f, 0.0f, -1.0f, -1.0f));	// This is the near-plane, right? Because we only have to check against that. For tlbr planes of the frustum, we can just draw, and then clamp it because it's outside the screen
		vertices = clipPolygonToPlaneIn4D(vertices, Vec4f(0.0f, 0.0f, 1.0f, -1.0f));	// This is the near-plane, right? Because we only have to check against that. For tlbr planes of the frustum, we can just draw, and then clamp it because it's outside the screen
		//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(0.0f, 0.0f, 1.0f, -1.0f));
		//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(-1.0f, 0.0f, 0.0f, -1.0f));
		//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(1.0f, 0.0f, 0.0f, -1.0f));
		//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(0.0f, -1.0f, 0.0f, -1.0f));
		//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(0.0f, 1.0f, 0.0f, -1.0f));
		/* Note from mail: (note: stuff might flip because we change z/P-matrix?)
		vertices = clipPolygonToPlaneIn4D(vertices, Vec4f(0.0f, 0.0f, -1.0f, -1.0f));
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
	return make_pair(colorBuffer, depthBuffer);
}

boost::optional<TriangleToRasterize> SoftwareRenderer::processProspectiveTri(Vertex v0, Vertex v1, Vertex v2)
{
	TriangleToRasterize t;
	t.v0 = v0;	// no memcopy I think. the transformed vertices don't get copied and exist only once. They are a local variable in runVertexProcessor(), the ref is passed here, and if we need to rasterize it, it gets push_back'ed (=copied?) to trianglesToRasterize. Perfect I think. TODO: Not anymore, no ref here
	t.v1 = v1;
	t.v2 = v2;

	// Only for texturing or perspective texturing:
	//t.texture = _texture;
	t.one_over_z0 = 1.0 / (double)t.v0.position[3];
	t.one_over_z1 = 1.0 / (double)t.v1.position[3];
	t.one_over_z2 = 1.0 / (double)t.v2.position[3];

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
	t.v0.position[0] = (t.v0.position[0] + 1) * (viewportWidth / 2.0f); // TODO: We could use our clipToScreen...etc functions?
	t.v0.position[1] = (t.v0.position[1] + 1) * (viewportHeight / 2.0f);
	t.v0.position[1] = viewportHeight - t.v0.position[1];
	t.v1.position[0] = (t.v1.position[0] + 1) * (viewportWidth / 2.0f);
	t.v1.position[1] = (t.v1.position[1] + 1) * (viewportHeight / 2.0f);
	t.v1.position[1] = viewportHeight - t.v1.position[1];
	t.v2.position[0] = (t.v2.position[0] + 1) * (viewportWidth / 2.0f);
	t.v2.position[1] = (t.v2.position[1] + 1) * (viewportHeight / 2.0f);
	t.v2.position[1] = viewportHeight - t.v2.position[1];
	// My last SW-renderer was:
	// x_w = (x *  vW/2) + vW/2; // equivalent to above
	// y_w = (y * -vH/2) + vH/2; // equiv? Todo!
	// CG book says: (check!)
	// x_w = (x *  vW/2) + (vW-1)/2;
	// y_w = (y * -vH/2) + (vH-1)/2;

	if (doBackfaceCulling) {
		if (!utils::areVerticesCCWInScreenSpace(t.v0, t.v1, t.v2))
			return boost::none;
	}

	// Get the bounding box of the triangle:
	cv::Rect boundingBox = render::utils::calculateBoundingBox(t.v0, t.v1, t.v2, viewportWidth, viewportHeight);
	t.minX = boundingBox.x;
	t.maxX = boundingBox.x + boundingBox.width;
	t.minY = boundingBox.y;
	t.maxY = boundingBox.y + boundingBox.height;

	if (t.maxX <= t.minX || t.maxY <= t.minY) 	// Note: Can the width/height of the bbox be negative? Maybe we only need to check for equality here?
		return boost::none;

	// Which of these is for texturing, mipmapping, what for perspective?
	// for partial derivatives computation
	t.alphaPlane = plane(Vec3f(t.v0.position[0], t.v0.position[1], t.v0.texcrd[0] * t.one_over_z0),
		Vec3f(t.v1.position[0], t.v1.position[1], t.v1.texcrd[0] * t.one_over_z1),
		Vec3f(t.v2.position[0], t.v2.position[1], t.v2.texcrd[0] * t.one_over_z2));
	t.betaPlane = plane(Vec3f(t.v0.position[0], t.v0.position[1], t.v0.texcrd[1] * t.one_over_z0),
		Vec3f(t.v1.position[0], t.v1.position[1], t.v1.texcrd[1] * t.one_over_z1),
		Vec3f(t.v2.position[0], t.v2.position[1], t.v2.texcrd[1] * t.one_over_z2));
	t.gammaPlane = plane(Vec3f(t.v0.position[0], t.v0.position[1], t.one_over_z0),
		Vec3f(t.v1.position[0], t.v1.position[1], t.one_over_z1),
		Vec3f(t.v2.position[0], t.v2.position[1], t.one_over_z2));
	t.one_over_alpha_c = 1.0f / t.alphaPlane.c;
	t.one_over_beta_c = 1.0f / t.betaPlane.c;
	t.one_over_gamma_c = 1.0f / t.gammaPlane.c;
	t.alpha_ffx = -t.alphaPlane.a * t.one_over_alpha_c;
	t.beta_ffx = -t.betaPlane.a * t.one_over_beta_c;
	t.gamma_ffx = -t.gammaPlane.a * t.one_over_gamma_c;
	t.alpha_ffy = -t.alphaPlane.b * t.one_over_alpha_c;
	t.beta_ffy = -t.betaPlane.b * t.one_over_beta_c;
	t.gamma_ffy = -t.gammaPlane.b * t.one_over_gamma_c;

	// Use t
	return boost::optional<TriangleToRasterize>(t);
}

void SoftwareRenderer::rasterTriangle(TriangleToRasterize triangle)
{
	TriangleToRasterize t = triangle; // remove this, use 'triangle'
	for (int yi = t.minY; yi <= t.maxY; yi++)
	{
		for (int xi = t.minX; xi <= t.maxX; xi++)
		{
			// we want centers of pixels to be used in computations. TODO: Do we?
			float x = (float)xi + 0.5f;
			float y = (float)yi + 0.5f;

			// these will be used for barycentric weights computation
			t.one_over_v0ToLine12 = 1.0 / utils::implicitLine(t.v0.position[0], t.v0.position[1], t.v1.position, t.v2.position);
			t.one_over_v1ToLine20 = 1.0 / utils::implicitLine(t.v1.position[0], t.v1.position[1], t.v2.position, t.v0.position);
			t.one_over_v2ToLine01 = 1.0 / utils::implicitLine(t.v2.position[0], t.v2.position[1], t.v0.position, t.v1.position);
			// affine barycentric weights
			double alpha = utils::implicitLine(x, y, t.v1.position, t.v2.position) * t.one_over_v0ToLine12;
			double beta = utils::implicitLine(x, y, t.v2.position, t.v0.position) * t.one_over_v1ToLine20;
			double gamma = utils::implicitLine(x, y, t.v0.position, t.v1.position) * t.one_over_v2ToLine01;

			// if pixel (x, y) is inside the triangle or on one of its edges
			if (alpha >= 0 && beta >= 0 && gamma >= 0)
			{
				int pixelIndexRow = yi;
				int pixelIndexCol = xi;

				double z_affine = alpha*(double)t.v0.position[2] + beta*(double)t.v1.position[2] + gamma*(double)t.v2.position[2];
				// The '<= 1.0' clips against the far-plane in NDC. We clip against the near-plane earlier.
				if (z_affine < depthBuffer.at<double>(pixelIndexRow, pixelIndexCol)/* && z_affine <= 1.0*/)
				{
					// perspective-correct barycentric weights
					double d = alpha*t.one_over_z0 + beta*t.one_over_z1 + gamma*t.one_over_z2;
					d = 1.0 / d;
					alpha *= d*t.one_over_z0; // In case of affine cam matrix, everything is 1 and a/b/g don't get changed.
					beta *= d*t.one_over_z1;
					gamma *= d*t.one_over_z2;

					// attributes interpolation
					Vec3f color_persp = alpha*t.v0.color + beta*t.v1.color + gamma*t.v2.color;
					Vec2f texCoord_persp = alpha*t.v0.texcrd + beta*t.v1.texcrd + gamma*t.v2.texcrd;

					Vec3f pixelColor;
					// Pixel Shader:
					if (doTexturing) {	// We use texturing
						// check if texture != NULL?
						// partial derivatives (for mip-mapping)
						float u_over_z = -(t.alphaPlane.a*x + t.alphaPlane.b*y + t.alphaPlane.d) * t.one_over_alpha_c;
						float v_over_z = -(t.betaPlane.a*x + t.betaPlane.b*y + t.betaPlane.d) * t.one_over_beta_c;
						float one_over_z = -(t.gammaPlane.a*x + t.gammaPlane.b*y + t.gammaPlane.d) * t.one_over_gamma_c;
						float one_over_squared_one_over_z = 1.0f / pow(one_over_z, 2);

						dudx = one_over_squared_one_over_z * (t.alpha_ffx * one_over_z - u_over_z * t.gamma_ffx);
						dudy = one_over_squared_one_over_z * (t.beta_ffx * one_over_z - v_over_z * t.gamma_ffx);
						dvdx = one_over_squared_one_over_z * (t.alpha_ffy * one_over_z - u_over_z * t.gamma_ffy);
						dvdy = one_over_squared_one_over_z * (t.beta_ffy * one_over_z - v_over_z * t.gamma_ffy);

						dudx *= currentTexture->mipmaps[0].cols;
						dudy *= currentTexture->mipmaps[0].cols;
						dvdx *= currentTexture->mipmaps[0].rows;
						dvdy *= currentTexture->mipmaps[0].rows;

						// The Texture is in BGR, thus tex2D returns BGR
						Vec3f textureColor = tex2D(texCoord_persp); // uses the current texture
						pixelColor = Vec3f(textureColor[2], textureColor[1], textureColor[0]);
						// other: color.mul(tex2D(texture, texCoord));
						// Old note: for texturing, we load the texture as BGRA, so the colors get the wrong way in the next few lines...
					}
					else {	// We use vertex-coloring
						// color_persp is in RGB
						pixelColor = color_persp;
					}

					// clamp bytes to 255
					unsigned char red = (unsigned char)(255.0f * min(pixelColor[0], 1.0f)); // Todo: Proper casting (rounding?)
					unsigned char green = (unsigned char)(255.0f * min(pixelColor[1], 1.0f));
					unsigned char blue = (unsigned char)(255.0f * min(pixelColor[2], 1.0f));

					// update buffers
					colorBuffer.at<Vec4b>(pixelIndexRow, pixelIndexCol)[0] = blue;
					colorBuffer.at<Vec4b>(pixelIndexRow, pixelIndexCol)[1] = green;
					colorBuffer.at<Vec4b>(pixelIndexRow, pixelIndexCol)[2] = red;
					colorBuffer.at<Vec4b>(pixelIndexRow, pixelIndexCol)[3] = 255; // alpha, or 1.0f?
					depthBuffer.at<double>(pixelIndexRow, pixelIndexCol) = z_affine;
				}
			}
		}
	}
}

std::vector<Vertex> SoftwareRenderer::clipPolygonToPlaneIn4D(const std::vector<Vertex>& vertices, const Vec4f& planeNormal)
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
			Vec4f direction = vertices[b].position - vertices[a].position;
			float t = -(planeNormal.dot(vertices[a].position)) / (planeNormal.dot(direction));

			Vec4f position = vertices[a].position + t*direction;
			Vec3f color = vertices[a].color + t*(vertices[b].color - vertices[a].color);
			Vec2f texCoord = vertices[a].texcrd + t*(vertices[b].texcrd - vertices[a].texcrd);	// We could omit that if we don't render with texture.

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
}

Vec3f SoftwareRenderer::tex2D(const Vec2f& texCoord)
{
	return (1.0f / 255.0f) * tex2D_linear_mipmap_linear(texCoord);
}

Vec3f SoftwareRenderer::tex2D_linear_mipmap_linear(const Vec2f& texCoord)
{
	float px = std::sqrt(std::pow(dudx, 2) + std::pow(dvdx, 2));
	float py = std::sqrt(std::pow(dudy, 2) + std::pow(dvdy, 2));
	float lambda = std::log(max(px, py)) / CV_LOG2;
	unsigned char mipmapIndex1 = clamp((int)lambda, 0.0f, max(currentTexture->widthLog, currentTexture->heightLog) - 1);
	unsigned char mipmapIndex2 = mipmapIndex1 + 1;

	Vec2f imageTexCoord = texCoord_wrap(texCoord);
	Vec2f imageTexCoord1 = imageTexCoord;
	imageTexCoord1[0] *= currentTexture->mipmaps[mipmapIndex1].cols;
	imageTexCoord1[1] *= currentTexture->mipmaps[mipmapIndex1].rows;
	Vec2f imageTexCoord2 = imageTexCoord;
	imageTexCoord2[0] *= currentTexture->mipmaps[mipmapIndex2].cols;
	imageTexCoord2[1] *= currentTexture->mipmaps[mipmapIndex2].rows;

	Vec3f color, color1, color2;
	color1 = tex2D_linear(imageTexCoord1, mipmapIndex1);
	color2 = tex2D_linear(imageTexCoord2, mipmapIndex2);
	float lambdaFrac = max(lambda, 0.0f);
	lambdaFrac = lambdaFrac - (int)lambdaFrac;
	color = (1.0f - lambdaFrac)*color1 + lambdaFrac*color2;

	return color;
}

Vec2f SoftwareRenderer::texCoord_wrap(const Vec2f& texCoord)
{
	return Vec2f(texCoord[0] - (int)texCoord[0], texCoord[1] - (int)texCoord[1]);
}

Vec3f SoftwareRenderer::tex2D_linear(const Vec2f& imageTexCoord, unsigned char mipmapIndex)
{
	int x = (int)imageTexCoord[0];
	int y = (int)imageTexCoord[1];
	float alpha = imageTexCoord[0] - x;
	float beta = imageTexCoord[1] - y;
	float oneMinusAlpha = 1.0f - alpha;
	float oneMinusBeta = 1.0f - beta;
	float a = oneMinusAlpha * oneMinusBeta;
	float b = alpha * oneMinusBeta;
	float c = oneMinusAlpha * beta;
	float d = alpha * beta;
	Vec3f color;

	//int pixelIndex;
	//pixelIndex = getPixelIndex_wrap(x, y, texture->mipmaps[mipmapIndex].cols, texture->mipmaps[mipmapIndex].rows);
	int pixelIndexCol = x; if (pixelIndexCol == currentTexture->mipmaps[mipmapIndex].cols) { pixelIndexCol = 0; }
	int pixelIndexRow = y; if (pixelIndexRow == currentTexture->mipmaps[mipmapIndex].rows) { pixelIndexRow = 0; }
	//std::cout << texture->mipmaps[mipmapIndex].cols << " " << texture->mipmaps[mipmapIndex].rows << " " << texture->mipmaps[mipmapIndex].channels() << std::endl;
	//cv::imwrite("mm.png", texture->mipmaps[mipmapIndex]);
	color[0] = a * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] = a * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] = a * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	//pixelIndex = getPixelIndex_wrap(x + 1, y, texture->mipmaps[mipmapIndex].cols, texture->mipmaps[mipmapIndex].rows);
	pixelIndexCol = x + 1; if (pixelIndexCol == currentTexture->mipmaps[mipmapIndex].cols) { pixelIndexCol = 0; }
	pixelIndexRow = y; if (pixelIndexRow == currentTexture->mipmaps[mipmapIndex].rows) { pixelIndexRow = 0; }
	color[0] += b * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] += b * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] += b * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	//pixelIndex = getPixelIndex_wrap(x, y + 1, texture->mipmaps[mipmapIndex].cols, texture->mipmaps[mipmapIndex].rows);
	pixelIndexCol = x; if (pixelIndexCol == currentTexture->mipmaps[mipmapIndex].cols) { pixelIndexCol = 0; }
	pixelIndexRow = y + 1; if (pixelIndexRow == currentTexture->mipmaps[mipmapIndex].rows) { pixelIndexRow = 0; }
	color[0] += c * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] += c * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] += c * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	//pixelIndex = getPixelIndex_wrap(x + 1, y + 1, texture->mipmaps[mipmapIndex].cols, texture->mipmaps[mipmapIndex].rows);
	pixelIndexCol = x + 1; if (pixelIndexCol == currentTexture->mipmaps[mipmapIndex].cols) { pixelIndexCol = 0; }
	pixelIndexRow = y + 1; if (pixelIndexRow == currentTexture->mipmaps[mipmapIndex].rows) { pixelIndexRow = 0; }
	color[0] += d * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] += d * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] += d * currentTexture->mipmaps[mipmapIndex].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	return color;
}

float SoftwareRenderer::clamp(float x, float a, float b)
{
	return max(min(x, b), a);
}

Vec3f SoftwareRenderer::projectVertex(Vec4f vertex, Mat mvp)
{
	Mat clipSpace = mvp * Mat(vertex);
	Vec4f clipSpaceV(clipSpace);
	// divide by w
	clipSpaceV = clipSpaceV / clipSpaceV[3];

	// project from 4D to 2D window position with depth value in z coordinate
	// Viewport transform:
	clipSpaceV[0] = (clipSpaceV[0] + 1) * (viewportWidth / 2.0f);
	clipSpaceV[1] = (clipSpaceV[1] + 1) * (viewportHeight / 2.0f);
	clipSpaceV[1] = viewportHeight - clipSpaceV[1];

	// Find the correct z-value for the exact pixel the vertex is landing in.
	// We need this to get the same depth value for the vertex than the one in the z-buffer.
	// No, this doesn't work, we're only projecting a vertex, not a triangle => no barycentric coords
/*	int xi = cvRound(clipSpaceV[0]);
	int yi = cvRound(clipSpaceV[1]);
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
		int pixelIndexRow = yi;
		int pixelIndexCol = xi;

		double z_affine = alpha*(double)t.v0.position[2] + beta*(double)t.v1.position[2] + gamma*(double)t.v2.position[2];
		// The '<= 1.0' clips against the far-plane in NDC. We clip against the near-plane earlier.
		if (z_affine < depthBuffer.at<double>(pixelIndexRow, pixelIndexCol) && z_affine <= 1.0)
	*/

	return Vec3f(clipSpaceV[0], clipSpaceV[1], clipSpaceV[2]);
}

} /* namespace render */
