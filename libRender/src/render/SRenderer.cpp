/*!
 * \file SRenderer.cpp
 *
 * \author Patrik Huber
 * \date December 4, 2012
 *
 * [comment here]
 */

#include "render/SRenderer.hpp"
#include "render/MatrixUtils.hpp"
#include "render/Texture.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cmath>
#include <iostream>
#include <array>
#include <tuple>

namespace render {

SRenderer::SRenderer(void)
{
}


SRenderer::~SRenderer(void)
{
}

SRenderer* SRenderer::Instance(void)
{
	static SRenderer instance;
	return &instance;
}


void SRenderer::create()
{
	setViewport(640, 480);
	camera.init();

	this->colorBuffer = cv::Mat::zeros(screenHeight, screenWidth, CV_8UC4);
	this->depthBuffer = cv::Mat::ones(screenHeight, screenWidth, CV_64FC1)*1000000;

}

void SRenderer::destroy()
{
}

void SRenderer::setViewport(unsigned int screenWidth, unsigned int screenHeight)
{
	this->screenWidth = screenWidth;
	this->screenHeight = screenHeight;

	/*this->windowTransform = (cv::Mat_<float>(4,4) << 
						  (float)screenWidth/2.0f,		 0.0f,						0.0f,	0.0f,
						  0.0f,							 (float)screenHeight/2.0f,	0.0f,	0.0f,
						  0.0f,							 0.0f,						1.0f,	0.0f,
						  (float)screenWidth/2.0f,		 (float)screenHeight/2.0f,	0.0f,	1.0f);*/	//<-- original
	this->windowTransform = (cv::Mat_<float>(4,4) << 
		(float)screenWidth/2.0f,		0.0f,						0.0f,	(float)screenWidth/2.0f, // CG book says (screenWidth-1)/2.0f for second value?
		0.0f,							(float)screenHeight/2.0f,	0.0f,	(float)screenHeight/2.0f,
		0.0f,							0.0f,						1.0f,	0.0f,
		0.0f,							0.0f,						0.0f,	1.0f);

}

cv::Mat SRenderer::constructViewTransform()
{
	cv::Mat translate = render::utils::MatrixUtils::createTranslationMatrix(-camera.getEye()[0], -camera.getEye()[1], -camera.getEye()[2]);

	cv::Mat rotate = (cv::Mat_<float>(4,4) << 
				   camera.getRightVector()[0],		camera.getRightVector()[1],		camera.getRightVector()[2],		0.0f,
			       camera.getUpVector()[0],			camera.getUpVector()[1],		camera.getUpVector()[2],		0.0f,
			       -camera.getForwardVector()[0],	-camera.getForwardVector()[1],	-camera.getForwardVector()[2],	0.0f,
			       0.0f,				0.0f,				0.0f,				1.0f);

	return rotate * translate;
}

cv::Mat SRenderer::constructProjTransform()
{
	cv::Mat perspective = (cv::Mat_<float>(4,4) << 
						camera.frustum.n,	0.0f,				0.0f,									0.0f,
						0.0f,				camera.frustum.n,	0.0f,									0.0f,
						0.0f,				0.0f,				camera.frustum.n + camera.frustum.f,	camera.frustum.n * camera.frustum.f, // CG book has -f*n ? (I had +f*n before)
						0.0f,				0.0f,				-1.0f, /* CG has +1 here, I had -1 */	0.0f);
	
	cv::Mat orthogonal = (cv::Mat_<float>(4,4) << 
		2.0f / (camera.frustum.r - camera.frustum.l),	0.0f,											0.0f,											-(camera.frustum.r + camera.frustum.l) / (camera.frustum.r - camera.frustum.l),
		0.0f,											2.0f / (camera.frustum.t - camera.frustum.b),	0.0f,											-(camera.frustum.t + camera.frustum.b) / (camera.frustum.t - camera.frustum.b),
		0.0f,											0.0f,											2.0f / (camera.frustum.n - camera.frustum.f),	-(camera.frustum.n + camera.frustum.f) / (camera.frustum.f - camera.frustum.n), // CG book has denominator (n-f) ? I had (f-n) before.
		0.0f,											0.0f,											0.0f,											1.0f);
	
	return orthogonal * perspective;
}

void SRenderer::setMesh(const Mesh* mesh) // make an optional argument, an index-list of vertices or so to render, if we don't want to render the full mesh.
{
	currentMesh = mesh;
}

void SRenderer::setTransform(const cv::Mat transform)
{
	currentTransform = transform;
}

void SRenderer::draw(ushort trianglesNum)
{
	DrawCall drawCall;

	drawCall.mesh = currentMesh;
	drawCall.trianglesNum = (trianglesNum == 0 ? currentMesh->tvi.size() : trianglesNum);	// not needed? only if I don't draw the whole mesh?
	drawCall.transform = currentTransform;
	if(currentMesh->hasTexture==false) {
		drawCall.texture = NULL;
	} else {
		drawCall.texture = currentTexture;
	}

	drawCalls.push_back(drawCall);
}

void SRenderer::end()
{
	runVertexProcessor();
	runPixelProcessor();
	
	drawCalls.clear();
	trianglesToRasterize.clear();
}

void SRenderer::setTexture(const Texture& texture)
{
	currentTexture = &texture;
}

void SRenderer::runVertexProcessor()
{
	for (unsigned int i = 0; i < drawCalls.size(); i++)
	{
		for (unsigned int j = 0; j < drawCalls[i].trianglesNum; j++)	// loops through mesh.tvi
		{
			Triangle triangle;	// this is going to be the triangle in clip coordinates (after MVP-transform)

			for (unsigned char k = 0; k < 3; k++) {	// I should transform all at once. Vertices = columns ( [V1; V2; V3; ...] * ViewProj... )
				triangle.vertex[k] = runVertexShader( drawCalls[i].mesh, drawCalls[i].transform, drawCalls[i].mesh->tvi[j][k] );	// if tvi != tci then of course the vertex-coloring is wrong!
				// vertex data gets copied here. (created also in the function). Could avoid this by using pointer, ref... whatever...
				//triangle.vertex[k] = runVertexShader(drawCalls[i].transform, drawCalls[i].trianglesBuffer[j].vertices[k]);
			}

			// Hmm I transform many vertices twice or even more, because they are used in more than one triangle.
			// Maybe make that better, eg a loop over mesh.vertices instead of tvi, or maybe make a new list with all transformed triangles, in same order, then use this plus mesh.tvi
			// this way I can also transform them all at once!

			// classify vertices visibility with respect to the planes of the view frustum
			unsigned char visibilityBits[3];

			for (unsigned char k = 0; k < 3; k++)
			{
				visibilityBits[k] = 0;

				if (triangle.vertex[k].position[0] < -triangle.vertex[k].position[3])
					visibilityBits[k] |= 1;
				if (triangle.vertex[k].position[0] > triangle.vertex[k].position[3])
					visibilityBits[k] |= 2;
				if (triangle.vertex[k].position[1] < -triangle.vertex[k].position[3])
					visibilityBits[k] |= 4;
				if (triangle.vertex[k].position[1] > triangle.vertex[k].position[3])
					visibilityBits[k] |= 8;
				if (triangle.vertex[k].position[2] < -triangle.vertex[k].position[3])
					visibilityBits[k] |= 16;
				if (triangle.vertex[k].position[2] > triangle.vertex[k].position[3])
					visibilityBits[k] |= 32;
			}

			// all vertices are not visible - reject the triangle
			if ( (visibilityBits[0] & visibilityBits[1] & visibilityBits[2]) > 0 )
			{
				continue;
			}

			// all vertices are visible - pass the whole triangle to the rasterizer
			if ( (visibilityBits[0] | visibilityBits[1] | visibilityBits[2]) == 0 )
			{
				processProspectiveTriangleToRasterize(	// does nothing with the texture, only copy either NULL or the texcoords
					triangle.vertex[0],
					triangle.vertex[1],
					triangle.vertex[2],
					drawCalls[i].texture);

				continue;
			}

			// at this moment the triangle is known to be intersecting one of the view frustum's planes

			std::vector<Vertex> vertices;
			vertices.push_back(triangle.vertex[0]);
			vertices.push_back(triangle.vertex[1]);
			vertices.push_back(triangle.vertex[2]);

			vertices = clipPolygonToPlaneIn4D(vertices, cv::Vec4f(0.0f, 0.0f, -1.0f, -1.0f));	// This is the near-plane, right? Because we only have to check against that. For tlbr planes of the frustum, we can just draw, and then clamp it because it's outside the screen
			//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(0.0f, 0.0f, 1.0f, -1.0f));
			//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(-1.0f, 0.0f, 0.0f, -1.0f));
			//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(1.0f, 0.0f, 0.0f, -1.0f));
			//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(0.0f, -1.0f, 0.0f, -1.0f));
			//	vertices = clipPolygonToPlaneIn4D(vertices, vec4(0.0f, 1.0f, 0.0f, -1.0f));

			// triangulation of the polygon formed of vertices array
			if (vertices.size() >= 3)
			{
				for (unsigned char k = 0; k < vertices.size() - 2; k++)
				{
					processProspectiveTriangleToRasterize(
						vertices[0],
						vertices[1 + k],
						vertices[2 + k],
						drawCalls[i].texture);
				}
			}
		}
	}
}

Vertex SRenderer::runVertexShader(const Mesh* mesh, const cv::Mat& transform, const int vertexNum)
{
	Vertex output;
	
	cv::Mat tmp =  transform * cv::Mat(mesh->vertex[vertexNum].position);	// places the vec as a row in the matrix. This is viewProjTransform * worldTransform
	output.position[0] = tmp.at<float>(0, 0);
	output.position[1] = tmp.at<float>(1, 0);
	output.position[2] = tmp.at<float>(2, 0);
	output.position[3] = tmp.at<float>(3, 0);

	output.color = mesh->vertex[vertexNum].color;
	if(mesh->hasTexture)	// not the 100% fastest/most correct approach. What if the model has a texture but we want it to render with vertex color, etc.
		output.texcrd = mesh->vertex[vertexNum].texcrd;

	return output;
}

void SRenderer::processProspectiveTriangleToRasterize(const Vertex& _v0, const Vertex& _v1, const Vertex& _v2, const Texture* _texture)
{
	TriangleToRasterize t;

	t.v0 = _v0;	// no memcopy I think. the transformed vertices don't get copied and exist only once. They are a local variable in runVertexProcessor(), the ref is passed here, and if we need to rasterize it, it gets push_back'ed (=copied?) to trianglesToRasterize. Perfect I think.
	t.v1 = _v1;
	t.v2 = _v2;
	t.texture = _texture;

	t.one_over_z0 = 1.0 / (double)t.v0.position[3];
	t.one_over_z1 = 1.0 / (double)t.v1.position[3];
	t.one_over_z2 = 1.0 / (double)t.v2.position[3];

	// project from 4D to 2D window position with depth value in z coordinate
	t.v0.position = t.v0.position / t.v0.position[3];	// divide by w
	cv::Mat tmp = windowTransform * cv::Mat(t.v0.position);	// places the vec as a column in the matrix
	t.v0.position[0] = tmp.at<float>(0, 0);
	t.v0.position[1] = tmp.at<float>(1, 0);
	t.v0.position[2] = tmp.at<float>(2, 0);
	t.v0.position[3] = tmp.at<float>(3, 0);
	
	t.v1.position = t.v1.position / t.v1.position[3];
	tmp = windowTransform * cv::Mat(t.v1.position);	// places the vec as a column in the matrix
	t.v1.position[0] = tmp.at<float>(0, 0);
	t.v1.position[1] = tmp.at<float>(1, 0);
	t.v1.position[2] = tmp.at<float>(2, 0);
	t.v1.position[3] = tmp.at<float>(3, 0);

	t.v2.position = t.v2.position / t.v2.position[3];
	tmp = windowTransform * cv::Mat(t.v2.position);	// places the vec as a column in the matrix
	t.v2.position[0] = tmp.at<float>(0, 0);
	t.v2.position[1] = tmp.at<float>(1, 0);
	t.v2.position[2] = tmp.at<float>(2, 0);
	t.v2.position[3] = tmp.at<float>(3, 0);

	if (!areVerticesCCWInScreenSpace(t.v0, t.v1, t.v2))
		return;

	// find bounding box for the triangle
	t.minX = std::max(std::min(t.v0.position[0], std::min(t.v1.position[0], t.v2.position[0])), 0.0f);
	t.maxX = std::min(std::max(t.v0.position[0], std::max(t.v1.position[0], t.v2.position[0])), (float)(screenWidth - 1));
	t.minY = std::max(std::min(t.v0.position[1], std::min(t.v1.position[1], t.v2.position[1])), 0.0f);
	t.maxY = std::min(std::max(t.v0.position[1], std::max(t.v1.position[1], t.v2.position[1])), (float)(screenHeight - 1));

	if (t.maxX <= t.minX || t.maxY <= t.minY)
		return;

	// these will be used for barycentric weights computation
	t.one_over_v0ToLine12 = 1.0 / implicitLine(t.v0.position[0], t.v0.position[1], t.v1.position, t.v2.position);
	t.one_over_v1ToLine20 = 1.0 / implicitLine(t.v1.position[0], t.v1.position[1], t.v2.position, t.v0.position);
	t.one_over_v2ToLine01 = 1.0 / implicitLine(t.v2.position[0], t.v2.position[1], t.v0.position, t.v1.position);

	// for partial derivatives computation
	t.alphaPlane = plane(cv::Vec3f(t.v0.position[0], t.v0.position[1], t.v0.texcrd[0]*t.one_over_z0),
						 cv::Vec3f(t.v1.position[0], t.v1.position[1], t.v1.texcrd[0]*t.one_over_z1),
						 cv::Vec3f(t.v2.position[0], t.v2.position[1], t.v2.texcrd[0]*t.one_over_z2));
	t.betaPlane = plane(cv::Vec3f(t.v0.position[0], t.v0.position[1], t.v0.texcrd[1]*t.one_over_z0),
						cv::Vec3f(t.v1.position[0], t.v1.position[1], t.v1.texcrd[1]*t.one_over_z1),
						cv::Vec3f(t.v2.position[0], t.v2.position[1], t.v2.texcrd[1]*t.one_over_z2));
	t.gammaPlane = plane(cv::Vec3f(t.v0.position[0], t.v0.position[1], t.one_over_z0),
						 cv::Vec3f(t.v1.position[0], t.v1.position[1], t.one_over_z1),
						 cv::Vec3f(t.v2.position[0], t.v2.position[1], t.one_over_z2));
	t.one_over_alpha_c = 1.0f / t.alphaPlane.c;
	t.one_over_beta_c = 1.0f / t.betaPlane.c;
	t.one_over_gamma_c = 1.0f / t.gammaPlane.c;
	t.alpha_ffx = -t.alphaPlane.a * t.one_over_alpha_c;
	t.beta_ffx = -t.betaPlane.a * t.one_over_beta_c;
	t.gamma_ffx = -t.gammaPlane.a * t.one_over_gamma_c;
	t.alpha_ffy = -t.alphaPlane.b * t.one_over_alpha_c;
	t.beta_ffy = -t.betaPlane.b * t.one_over_beta_c;
	t.gamma_ffy = -t.gammaPlane.b * t.one_over_gamma_c;

	t.tileMinX = t.minX / 16;
	t.tileMinY = t.minY / 16;
	t.tileMaxX = t.maxX / 16;
	t.tileMaxY = t.maxY / 16;

	trianglesToRasterize.push_back(t);
}

std::vector<Vertex> SRenderer::clipPolygonToPlaneIn4D(const std::vector<Vertex>& vertices, const cv::Vec4f& planeNormal)
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

		if ( (fa < 0 && fb > 0) || (fa > 0 && fb < 0) )
		{
			cv::Vec4f direction = vertices[b].position - vertices[a].position;
			float t = - (planeNormal.dot(vertices[a].position)) / (planeNormal.dot(direction));

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
}

bool SRenderer::areVerticesCCWInScreenSpace(const Vertex& v0, const Vertex& v1, const Vertex& v2)
{
	float dx01 = v1.position[0] - v0.position[0];
	float dy01 = v1.position[1] - v0.position[1];
	float dx02 = v2.position[0] - v0.position[0];
	float dy02 = v2.position[1] - v0.position[1];

	return (dx01*dy02 - dy01*dx02 > 0.0f);
}

double SRenderer::implicitLine(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2)
{
	return ((double)v1[1] - (double)v2[1])*(double)x + ((double)v2[0] - (double)v1[0])*(double)y + (double)v1[0]*(double)v2[1] - (double)v2[0]*(double)v1[1];
}


void SRenderer::runPixelProcessor()
{
	// clear buffers
	this->colorBuffer = cv::Mat::zeros(screenHeight, screenWidth, CV_8UC4);
	this->depthBuffer = cv::Mat::ones(screenHeight, screenWidth, CV_64FC1)*1000000;

	for (unsigned int i = 0; i < trianglesToRasterize.size(); i++)
	{
		TriangleToRasterize& t = trianglesToRasterize[i];

		for (int yi = t.minY; yi <= t.maxY; yi++)
		{
			for (int xi = t.minX; xi <= t.maxX; xi++)
			{
				// we want centers of pixels to be used in computations
				float x = (float)xi + 0.5f;
				float y = (float)yi + 0.5f;

				// affine barycentric weights
				double alpha = implicitLine(x, y, t.v1.position, t.v2.position) * t.one_over_v0ToLine12;
				double beta = implicitLine(x, y, t.v2.position, t.v0.position) * t.one_over_v1ToLine20;
				double gamma = implicitLine(x, y, t.v0.position, t.v1.position) * t.one_over_v2ToLine01;

				// if pixel (x, y) is inside the triangle or on one of its edges
				if (alpha >= 0 && beta >= 0 && gamma >= 0)
				{
					//int pixelIndex = (screenHeight - 1 - yi)*screenWidth + xi;
					int pixelIndexRow = (screenHeight - 1 - yi);
					int pixelIndexCol = xi;
					
					double z_affine = alpha*(double)t.v0.position[2] + beta*(double)t.v1.position[2] + gamma*(double)t.v2.position[2];	// z

					if (z_affine < depthBuffer.at<double>(pixelIndexRow, pixelIndexCol) && z_affine <= 1.0)
					{
						// perspective-correct barycentric weights
						double d = alpha*t.one_over_z0 + beta*t.one_over_z1 + gamma*t.one_over_z2;
						d = 1.0 / d;
						alpha *= d*t.one_over_z0;
						beta *= d*t.one_over_z1;
						gamma *= d*t.one_over_z2;

						// attributes interpolation
						cv::Vec3f color_persp = alpha*t.v0.color + beta*t.v1.color + gamma*t.v2.color;
						cv::Vec2f texCoord_persp = alpha*t.v0.texcrd + beta*t.v1.texcrd + gamma*t.v2.texcrd;

						cv::Vec3f pixelColor;
						// partial derivatives (for mip-mapping)
						if(t.texture!=NULL) {	// We use texturing
							float u_over_z = -(t.alphaPlane.a*x + t.alphaPlane.b*y + t.alphaPlane.d) * t.one_over_alpha_c;
							float v_over_z = -(t.betaPlane.a*x + t.betaPlane.b*y + t.betaPlane.d) * t.one_over_beta_c;
							float one_over_z = -(t.gammaPlane.a*x + t.gammaPlane.b*y + t.gammaPlane.d) * t.one_over_gamma_c;
							float one_over_squared_one_over_z = 1.0f / pow(one_over_z, 2);

							dudx = one_over_squared_one_over_z * (t.alpha_ffx * one_over_z - u_over_z * t.gamma_ffx);
							dudy = one_over_squared_one_over_z * (t.beta_ffx * one_over_z - v_over_z * t.gamma_ffx);
							dvdx = one_over_squared_one_over_z * (t.alpha_ffy * one_over_z - u_over_z * t.gamma_ffy);
							dvdy = one_over_squared_one_over_z * (t.beta_ffy * one_over_z - v_over_z * t.gamma_ffy);

							dudx *= t.texture->mipmaps[0].cols;
							dudy *= t.texture->mipmaps[0].cols;
							dvdx *= t.texture->mipmaps[0].rows;
							dvdy *= t.texture->mipmaps[0].rows;

							// run pixel shader
							pixelColor = runPixelShader(t.texture, color_persp, texCoord_persp);
						} else {	// We use vertex-coloring
							pixelColor = runPixelShader(t.texture, color_persp, texCoord_persp, false);
						}

						// clamp bytes to 255
						unsigned char blue = (unsigned char)(255.0f * std::min(pixelColor[0], 1.0f));
						unsigned char green = (unsigned char)(255.0f * std::min(pixelColor[1], 1.0f));
						unsigned char red = (unsigned char)(255.0f * std::min(pixelColor[2], 1.0f));

						// update buffers
						//colorBuffer[4*pixelIndex + 0] = blue;
						colorBuffer.at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[0] = blue;
						//colorBuffer[4*pixelIndex + 1] = green;
						colorBuffer.at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[1] = green;
						//colorBuffer[4*pixelIndex + 2] = red;
						colorBuffer.at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[2] = red;
						//depthBuffer[pixelIndex] = z_affine;
						depthBuffer.at<double>(pixelIndexRow, pixelIndexCol) = z_affine;
					}
				}
			}
		}
	}

}

cv::Vec3f SRenderer::runPixelShader(const Texture* texture, const cv::Vec3f& color, const cv::Vec2f& texCoord, bool useTexturing)
{
	if(useTexturing) {
		return color.mul(tex2D(texture, texCoord));	// element-wise multiplication
	} else {
		return color;
	}
}

cv::Vec3f SRenderer::tex2D(const Texture* texture, const cv::Vec2f& texCoord)
{
	return (1.0f / 255.0f) * tex2D_linear_mipmap_linear(texture, texCoord);
}

cv::Vec3f SRenderer::tex2D_linear_mipmap_linear(const Texture* texture, const cv::Vec2f& texCoord)
{
	float px = std::sqrt(std::pow(dudx, 2) + std::pow(dvdx, 2));
	float py = std::sqrt(std::pow(dudy, 2) + std::pow(dvdy, 2));
	float lambda = std::log(std::max(px, py))/CV_LOG2;
	unsigned char mipmapIndex1 = clamp((int)lambda, 0.0f, std::max(texture->widthLog, texture->heightLog) - 1);
	unsigned char mipmapIndex2 = mipmapIndex1 + 1;

	cv::Vec2f imageTexCoord = texCoord_wrap(texCoord);
	cv::Vec2f imageTexCoord1 = imageTexCoord;
	imageTexCoord1[0] *= texture->mipmaps[mipmapIndex1].cols;
	imageTexCoord1[1] *= texture->mipmaps[mipmapIndex1].rows;
	cv::Vec2f imageTexCoord2 = imageTexCoord;
	imageTexCoord2[0] *= texture->mipmaps[mipmapIndex2].cols;
	imageTexCoord2[1] *= texture->mipmaps[mipmapIndex2].rows;

	cv::Vec3f color, color1, color2;
	color1 = tex2D_linear(texture, imageTexCoord1, mipmapIndex1);
	color2 = tex2D_linear(texture, imageTexCoord2, mipmapIndex2);
	float lambdaFrac = std::max(lambda, 0.0f);
	lambdaFrac = lambdaFrac - (int)lambdaFrac;
	color = (1.0f - lambdaFrac)*color1 + lambdaFrac*color2;

	return color;
}

cv::Vec2f SRenderer::texCoord_wrap(const cv::Vec2f& texCoord)
{
	return cv::Vec2f(texCoord[0]-(int)texCoord[0], texCoord[1]-(int)texCoord[1]);
}

cv::Vec3f SRenderer::tex2D_linear(const Texture* texture, const cv::Vec2f& imageTexCoord, unsigned char mipmapIndex)
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
	cv::Vec3f color;

	//int pixelIndex;
	//pixelIndex = getPixelIndex_wrap(x, y, texture->mipmaps[mipmapIndex].cols, texture->mipmaps[mipmapIndex].rows);
	int pixelIndexCol = x; if(pixelIndexCol==texture->mipmaps[mipmapIndex].cols) { pixelIndexCol = 0; }
	int pixelIndexRow = y; if(pixelIndexRow==texture->mipmaps[mipmapIndex].rows) { pixelIndexRow = 0; }
	//std::cout << texture->mipmaps[mipmapIndex].cols << " " << texture->mipmaps[mipmapIndex].rows << " " << texture->mipmaps[mipmapIndex].channels() << std::endl;
	//cv::imwrite("mm.png", texture->mipmaps[mipmapIndex]);
	color[0] = a * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] = a * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] = a * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	//pixelIndex = getPixelIndex_wrap(x + 1, y, texture->mipmaps[mipmapIndex].cols, texture->mipmaps[mipmapIndex].rows);
	pixelIndexCol = x+1; if(pixelIndexCol==texture->mipmaps[mipmapIndex].cols) { pixelIndexCol = 0; }
	pixelIndexRow = y; if(pixelIndexRow==texture->mipmaps[mipmapIndex].rows) { pixelIndexRow = 0; }
	color[0] += b * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] += b * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] += b * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	//pixelIndex = getPixelIndex_wrap(x, y + 1, texture->mipmaps[mipmapIndex].cols, texture->mipmaps[mipmapIndex].rows);
	pixelIndexCol = x; if(pixelIndexCol==texture->mipmaps[mipmapIndex].cols) { pixelIndexCol = 0; }
	pixelIndexRow = y+1; if(pixelIndexRow==texture->mipmaps[mipmapIndex].rows) { pixelIndexRow = 0; }
	color[0] += c * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] += c * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] += c * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	//pixelIndex = getPixelIndex_wrap(x + 1, y + 1, texture->mipmaps[mipmapIndex].cols, texture->mipmaps[mipmapIndex].rows);
	pixelIndexCol = x+1; if(pixelIndexCol==texture->mipmaps[mipmapIndex].cols) { pixelIndexCol = 0; }
	pixelIndexRow = y+1; if(pixelIndexRow==texture->mipmaps[mipmapIndex].rows) { pixelIndexRow = 0; }
	color[0] += d * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] += d * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] += d * texture->mipmaps[mipmapIndex].at<cv::Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	return color;
}

float SRenderer::clamp(float x, float a, float b)
{
	return std::max(std::min(x, b), a);
}

cv::Mat SRenderer::getRendererImage()
{
	cv::Mat res;
	cv::cvtColor(this->colorBuffer, res, CV_BGRA2BGR);	// good? does imwrite really don't write alpha-chan images? But maybe the alpha-chan isn't implemented very good anyway
	return res;
}

cv::Mat SRenderer::getRendererDepthBuffer()
{
	cv::Mat db = this->depthBuffer * 255.0;
	db.convertTo(db, CV_8UC1);
	return db;
}

}