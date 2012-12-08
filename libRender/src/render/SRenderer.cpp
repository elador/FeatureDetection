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

	//screenWidth_tiles = (screenWidth + 15) / 16;
	//screenHeight_tiles = (screenHeight + 15) / 16;

	colorBuffer.resize(screenWidth * screenHeight);	// or .reserve?
	depthBuffer.resize(screenWidth * screenHeight);	// or .reserve?

}

void SRenderer::destroy()
{
}

void SRenderer::setViewport(unsigned int screenWidth, unsigned int screenHeight)
{
	this->screenWidth = screenWidth;
	this->screenHeight = screenHeight;

	this->windowTransform = (cv::Mat_<float>(4,4) << 
						  (float)screenWidth/2.0f,		 0.0f,						0.0f,	0.0f,
						  0.0f,							 (float)screenHeight/2.0f,	0.0f,	0.0f,
						  0.0f,							 0.0f,						1.0f,	0.0f,
						  (float)screenWidth/2.0f,		 (float)screenHeight/2.0f,	0.0f,	1.0f);
	// order: M(row, col);

}

cv::Mat SRenderer::constructViewTransform(const cv::Vec3f& position, const cv::Vec3f& rightVector, const cv::Vec3f& upVector, const cv::Vec3f& forwardVector)
{
	cv::Mat translate = render::utils::MatrixUtils::createTranslationMatrix(-position[0], -position[1], -position[2]);

	cv::Mat rotate = (cv::Mat_<float>(4,4) << 
				   rightVector[0],	upVector[0],		forwardVector[0],	0.0f,
			       rightVector[1],	upVector[1],		forwardVector[1],	0.0f,
			       rightVector[2],	upVector[2],		forwardVector[2],	0.0f,
			       0.0f,			0.0f,				0.0f,				1.0f);

	return translate * rotate;
}

cv::Mat SRenderer::constructProjTransform(float left, float right, float bottom, float top, float zNear, float zFar)
{
	cv::Mat perspective = (cv::Mat_<float>(4,4) << 
						zNear,	0.0f,	0.0f,			0.0f,
						0.0f,	zNear,	0.0f,			0.0f,
						0.0f,	0.0f,	zNear + zFar,	-1.0f,
						0.0f,	0.0f,	zNear * zFar,	0.0f);

	cv::Mat orthogonal = (cv::Mat_<float>(4,4) << 
					   2.0f / (right - left),				0.0f,								0.0f,								0.0f,
					   0.0f,								2.0f / (top - bottom),				0.0f,								0.0f,
					   0.0f,								0.0f,								-2.0f / (zFar - zNear),				0.0f,
					   -(right + left) / (right - left),	-(top + bottom) / (top - bottom),	-(zNear + zFar) / (zFar - zNear),	1.0f);
	
	return perspective * orthogonal;
}

void SRenderer::setTrianglesBuffer(const std::vector<Triangle> trianglesBuffer)
{
	currentTrianglesBuffer = trianglesBuffer;	// maybe "&trianglesBuffer" for better speed? how does std::vector behave?
}

void SRenderer::setTransform(const cv::Mat transform)
{
	currentTransform = transform;
}

void SRenderer::draw(ushort trianglesNum)
{
	DrawCall drawCall;

	drawCall.trianglesBuffer = currentTrianglesBuffer;
	drawCall.trianglesNum = (trianglesNum == 0 ? currentTrianglesBuffer.size() : trianglesNum);
	drawCall.transform = currentTransform;
	//drawCall.texture = currentTexture;

	drawCalls.push_back(drawCall);
}

void SRenderer::end()
{
	runVertexProcessor();
	//runPixelProcessor();
	
	drawCalls.clear();
	trianglesToRasterize.clear();


	//screenSurface;	// draw on screen, flip buffer (with double-buffering) or just draw the image
	// probably write the colorBuffer to a .png.
	
	
}

void SRenderer::runVertexProcessor()
{
	for (unsigned int i = 0; i < drawCalls.size(); i++)
	{
		for (unsigned int j = 0; j < drawCalls[i].trianglesNum; j++)
		{
			Triangle triangle;

			for (unsigned char k = 0; k < 3; k++)
				triangle.vertices[k] = runVertexShader(drawCalls[i].transform, drawCalls[i].trianglesBuffer[j].vertices[k]);

			// classify vertices visibility with respect to the planes of the view frustum

			unsigned char visibilityBits[3];

			for (unsigned char k = 0; k < 3; k++)
			{
				visibilityBits[k] = 0;

				if (triangle.vertices[k].position[0] < -triangle.vertices[k].position[3])
					visibilityBits[k] |= 1;
				if (triangle.vertices[k].position[0] > triangle.vertices[k].position[3])
					visibilityBits[k] |= 2;
				if (triangle.vertices[k].position[1] < -triangle.vertices[k].position[3])
					visibilityBits[k] |= 4;
				if (triangle.vertices[k].position[1] > triangle.vertices[k].position[3])
					visibilityBits[k] |= 8;
				if (triangle.vertices[k].position[2] < -triangle.vertices[k].position[3])
					visibilityBits[k] |= 16;
				if (triangle.vertices[k].position[2] > triangle.vertices[k].position[3])
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
				processProspectiveTriangleToRasterize(
					triangle.vertices[0],
					triangle.vertices[1],
					triangle.vertices[2],
					drawCalls[i].texture);

				continue;
			}

			// at this moment the triangle is known to be intersecting one of the view frustum's planes

			std::vector<Vertex> vertices;
			vertices.push_back(triangle.vertices[0]);
			vertices.push_back(triangle.vertices[1]);
			vertices.push_back(triangle.vertices[2]);

			vertices = clipPolygonToPlaneIn4D(vertices, cv::Vec4f(0.0f, 0.0f, -1.0f, -1.0f));
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

Vertex SRenderer::runVertexShader(const cv::Mat& transform, const Vertex& input)
{
	Vertex output;

	cv::Mat tmp = cv::Mat(input.position) * transform;	// places the vec as a row in the matrix
	output.position[0] = tmp.at<float>(0, 0);
	output.position[1] = tmp.at<float>(0, 1);
	output.position[2] = tmp.at<float>(0, 2);
	output.position[3] = tmp.at<float>(0, 3);

	output.color = input.color;
	output.texCoord = input.texCoord;

	return output;
}

void SRenderer::processProspectiveTriangleToRasterize(const Vertex& _v0, const Vertex& _v1, const Vertex& _v2, const Texture* _texture)
{
	TriangleToRasterize t;

	t.v0 = _v0;
	t.v1 = _v1;
	t.v2 = _v2;
	t.texture = _texture;

	t.one_over_z0 = 1.0f / t.v0.position[3];
	t.one_over_z1 = 1.0f / t.v1.position[3];
	t.one_over_z2 = 1.0f / t.v2.position[3];

	// project from 4D to 2D window position with depth value in z coordinate
	t.v0.position = t.v0.position / t.v0.position[3];	// divide by w
	cv::Mat tmp = cv::Mat(t.v0.position) * windowTransform;	// places the vec as a row in the matrix
	t.v0.position[0] = tmp.at<float>(0, 0);
	t.v0.position[1] = tmp.at<float>(0, 1);
	t.v0.position[2] = tmp.at<float>(0, 2);
	t.v0.position[3] = tmp.at<float>(0, 3);
	
	t.v1.position = t.v1.position / t.v1.position[3];
	tmp = cv::Mat(t.v1.position) * windowTransform;	// places the vec as a row in the matrix
	t.v1.position[0] = tmp.at<float>(0, 0);
	t.v1.position[1] = tmp.at<float>(0, 1);
	t.v1.position[2] = tmp.at<float>(0, 2);
	t.v1.position[3] = tmp.at<float>(0, 3);

	t.v2.position = t.v2.position / t.v2.position[3];
	tmp = cv::Mat(t.v2.position) * windowTransform;	// places the vec as a row in the matrix
	t.v2.position[0] = tmp.at<float>(0, 0);
	t.v2.position[1] = tmp.at<float>(0, 1);
	t.v2.position[2] = tmp.at<float>(0, 2);
	t.v2.position[3] = tmp.at<float>(0, 3);

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
	t.one_over_v0ToLine12 = 1.0f / implicitLine(t.v0.position[0], t.v0.position[1], t.v1.position, t.v2.position);
	t.one_over_v1ToLine20 = 1.0f / implicitLine(t.v1.position[0], t.v1.position[1], t.v2.position, t.v0.position);
	t.one_over_v2ToLine01 = 1.0f / implicitLine(t.v2.position[0], t.v2.position[1], t.v0.position, t.v1.position);

	// for partial derivatives computation
	t.alphaPlane = plane(cv::Vec3f(t.v0.position[0], t.v0.position[1], t.v0.texCoord[0]*t.one_over_z0),
						 cv::Vec3f(t.v1.position[0], t.v1.position[1], t.v1.texCoord[0]*t.one_over_z1),
						 cv::Vec3f(t.v2.position[0], t.v2.position[1], t.v2.texCoord[0]*t.one_over_z2));
	t.betaPlane = plane(cv::Vec3f(t.v0.position[0], t.v0.position[1], t.v0.texCoord[1]*t.one_over_z0),
						cv::Vec3f(t.v1.position[0], t.v1.position[1], t.v1.texCoord[1]*t.one_over_z1),
						cv::Vec3f(t.v2.position[0], t.v2.position[1], t.v2.texCoord[1]*t.one_over_z2));
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
			cv::Vec2f texCoord = vertices[a].texCoord + t*(vertices[b].texCoord - vertices[a].texCoord);

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

float SRenderer::implicitLine(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2)
{
	return (v1[1] - v2[1])*x + (v2[0] - v1[0])*y + v1[0]*v2[1] - v2[0]*v1[1];
}

}