/*
 * QOpenGLRenderer.cpp
 *
 *  Created on: 04.04.2014
 *      Author: Patrik Huber
 */

#include "render/QOpenGLRenderer.hpp"

#include "render/Mesh.hpp"
#include "render/MeshUtils.hpp"
#include "render/MatrixUtils.hpp"

#include "render/SoftwareRenderer2.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>

using std::vector;
using cv::Mat;

namespace render {

QOpenGLRenderer::QOpenGLRenderer(QOpenGLContext* qOpenGlContext) : qOpenGlContext(qOpenGlContext), m_program(0)
{
	// this is initialize():
	m_program = new QOpenGLShaderProgram(qOpenGlContext);
	m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
	m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
	m_program->link();
	m_posAttr = m_program->attributeLocation("posAttr");
	m_colAttr = m_program->attributeLocation("colAttr");
	m_texAttr = m_program->attributeLocation("texAttr");
	m_texWeightAttr = m_program->attributeLocation("texWeightAttr");
	m_matrixUniform = m_program->uniformLocation("matrix");

	glEnable(GL_TEXTURE_2D);
	//cv::Mat ocvimg = cv::imread("C:\\Users\\Patrik\\Documents\\GitHub\\img.png");
	//cv::Mat ocvimg = cv::imread("C:\\Users\\Patrik\\Documents\\GitHub\\isoRegistered3D_square.png");
	cv::Mat ocvimg = cv::imread("C:\\Users\\Patrik\\Documents\\GitHub\\img.png");
	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, texture);
	cv::cvtColor(ocvimg, ocvimg, CV_BGR2RGB);
	//cv::flip(ocvimg, ocvimg, 0); // Flip around the x-axis
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, ocvimg.cols, ocvimg.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, ocvimg.ptr(0));

	std::cout << "GL_SHADING_LANGUAGE_VERSION: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

	// Set nearest filtering mode for texture minification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// Set bilinear filtering mode for texture magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Wrap texture coordinates by repeating
	// f.ex. texture coordinate (1.1, 1.2) is same as (0.1, 0.2)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // GL_REPEAT
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// TODO: Put in d'tor:
	// glDeleteTextures(1, &texture);
}

void QOpenGLRenderer::render(render::Mesh mesh)
{
	//const qreal retinaScale = devicePixelRatio();
	glViewport(0, 0, viewportWidth * retinaScale, viewportHeight * retinaScale);

	//glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT/* | GL_STENCIL_BUFFER_BIT*/);

	// Enable depth buffer
	glEnable(GL_DEPTH_TEST); // not to init
	//glDepthFunc(GL_LEQUAL);                               // The Type Of Depth Test To Do
	//glClearDepthf(1.0f);
	//glDepthMask(GL_TRUE);
	//glDepthRange(0.0, 1.0);
	// Enable back face culling
	glEnable(GL_CULL_FACE); // not to init

	glEnable(GL_TEXTURE_2D);

	// Note: OpenGL: CCW triangles = front-facing.
	// Coord-axis: right = +x, up = +y, to back = -z, to front = +z

	m_program->bind();

	// Store the 3DMM vertices in GLfloat's
	// Todo: Measure time. Possible improvements: 1) See if Vec3f is in contiguous storage, if yes, maybe change my Vertex/Mesh structure.
	// 2) Maybe change the datatypes (see http://www.opengl.org/wiki/Vertex_Specification_Best_Practices 'Attribute sizes')
	vector<GLfloat> mmVertices;
	//render::Mesh mesh = morphableModel.getMean();
	mmVertices.clear();
	int numTriangles = mesh.tvi.size();
	for (int i = 0; i < numTriangles; ++i)
	{
		// First vertex x, y, z of the triangle
		const auto& triangle = mesh.tvi[i];
		mmVertices.push_back(mesh.vertex[triangle[0]].position[0]);
		mmVertices.push_back(mesh.vertex[triangle[0]].position[1]);
		mmVertices.push_back(mesh.vertex[triangle[0]].position[2]);
		// Second vertex x, y, z
		mmVertices.push_back(mesh.vertex[triangle[1]].position[0]);
		mmVertices.push_back(mesh.vertex[triangle[1]].position[1]);
		mmVertices.push_back(mesh.vertex[triangle[1]].position[2]);
		// Third vertex x, y, z
		mmVertices.push_back(mesh.vertex[triangle[2]].position[0]);
		mmVertices.push_back(mesh.vertex[triangle[2]].position[1]);
		mmVertices.push_back(mesh.vertex[triangle[2]].position[2]);
	}
	vector<GLfloat> mmColors;
	for (int i = 0; i < numTriangles; ++i)
	{
		// First vertex x, y, z of the triangle
		const auto& triangle = mesh.tci[i];
		mmColors.push_back(mesh.vertex[triangle[0]].color[0]);
		mmColors.push_back(mesh.vertex[triangle[0]].color[1]);
		mmColors.push_back(mesh.vertex[triangle[0]].color[2]);
		// Second vertex x, y, z
		mmColors.push_back(mesh.vertex[triangle[1]].color[0]);
		mmColors.push_back(mesh.vertex[triangle[1]].color[1]);
		mmColors.push_back(mesh.vertex[triangle[1]].color[2]);
		// Third vertex x, y, z
		mmColors.push_back(mesh.vertex[triangle[2]].color[0]);
		mmColors.push_back(mesh.vertex[triangle[2]].color[1]);
		mmColors.push_back(mesh.vertex[triangle[2]].color[2]);
	}
	vector<GLfloat> mmTex;
	for (int i = 0; i < numTriangles; ++i)
	{
		// First vertex u, v of the triangle
		const auto& triangle = mesh.tci[i]; // use tti?
		mmTex.push_back(mesh.vertex[triangle[0]].texcrd[0]);
		mmTex.push_back(mesh.vertex[triangle[0]].texcrd[1]);
		// Second vertex u, v
		mmTex.push_back(mesh.vertex[triangle[1]].texcrd[0]);
		mmTex.push_back(mesh.vertex[triangle[1]].texcrd[1]);
		// Third vertex u, v
		mmTex.push_back(mesh.vertex[triangle[2]].texcrd[0]);
		mmTex.push_back(mesh.vertex[triangle[2]].texcrd[1]);
	}

	float aspect = static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight);
	QMatrix4x4 matrix;
	//matrix.ortho(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 0.1f, 100.0f); // l r b t n f
	//matrix.ortho(-70.0f, 70.0f, -70.0f, 70.0f, 0.1f, 1000.0f);
	matrix.perspective(60, aspect, 0.1, 100.0);
	matrix.translate(0, 0, -2);
	matrix.rotate(45.0f, 1.0f, 0.0f, 0.0f);
	matrix.rotate(0.0f, 0.0f, 1.0f, 0.0f);
	//matrix.scale(0.009f);
	//matrix.scale(0.003f);

	m_program->setUniformValue(m_matrixUniform, matrix);

	glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, &mmVertices[0]); // vertices
	glVertexAttribPointer(m_colAttr, 3, GL_FLOAT, GL_FALSE, 0, &mmColors[0]); // colors
	glVertexAttribPointer(m_texAttr, 2, GL_FLOAT, GL_FALSE, 0, &mmTex[0]); // texCoords
	m_program->setAttributeValue(m_texWeightAttr, 1.0f);
	m_program->setUniformValue("texture", texture);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, texture);

	glEnableVertexAttribArray(m_posAttr);
	glEnableVertexAttribArray(m_colAttr);
	glEnableVertexAttribArray(m_texAttr);

	glDrawArrays(GL_TRIANGLES, 0, numTriangles * 3); // 6; (2 triangles) how many vertices to render

	glDisableVertexAttribArray(m_texAttr);
	glDisableVertexAttribArray(m_colAttr);
	glDisableVertexAttribArray(m_posAttr);

	m_program->release();

	// Get the framebuffer
	glDisable(GL_TEXTURE_2D);
	uchar col;
	cv::Mat colb = cv::Mat::zeros(viewportHeight, viewportWidth, CV_8UC4);
	glGetError();
	glReadPixels(0, 0, viewportWidth, viewportHeight, GL_RGBA, GL_UNSIGNED_BYTE, colb.ptr(0)); // Caution: For QtQuick I guess we might into trouble as other stuff is in the buffer as well. Also, it doesn't seem to work if the window is occluded by another window.
	cv::flip(colb, colb, 0); // 0 = flip around x-axis. OpenGL and glReadPixels have (0, 0) at the bottom-left, while OpenCV has it on the top-left.
	cvtColor(colb, colb, cv::COLOR_RGBA2BGRA); // The OpenGL framebuffer is RGBA, OpenCV assumes BGRA, so we need to flip it to make it right for OpenCV.
	GLenum err0 = glGetError();
	glGetError();

	// Test: Get the depth buffer (probably slow)
	float depth;
	cv::Mat qdepthBuffer = cv::Mat::zeros(viewportHeight, viewportWidth, CV_32FC1);
	glReadPixels(320, 240, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
	//glReadPixels(0, 0, viewportWidth, viewportHeight, GL_DEPTH_COMPONENT, GL_FLOAT, qdepthBuffer.ptr(0));
	GLenum err = glGetError();
	std::cout << err << std::endl;
	std::cout << depth << std::endl;
	int dbits;
	glGetIntegerv(GL_DEPTH_BITS, &dbits);
	std::cout << "dbits: " << dbits << std::endl;
	//glBindFramebuffer(GL_READ_FRAMEBUFFER, 0); ?
	//glPixelStorei(GL_PACK_ALIGNMENT, 1);?

	//cv::Mat framebuffer = cv::imread("C:\\Users\\Patrik\\Documents\\GitHub\\box_screenbuffer11.png");
	//cv::Mat textureMap = render::utils::MeshUtils::extractTexture(mesh, matrix, viewportWidth, viewportHeight, framebuffer);
	//cv::imwrite("C:\\Users\\Patrik\\Documents\\GitHub\\img_extracted11.png", textureMap);
	++m_frame;


	// SOFTWARE RENDERER START
	Mat framebuffer = Mat::zeros(viewportHeight, viewportWidth, CV_8UC3);

	// Prepare:
	// this is only the method how to draw (e.g. make tris and draw them (i.e. duplicate vertices), but in the end we have a VertexShader that operates per vertex
	for (const auto& triIndices : mesh.tvi) {
		//For every triangle: Like OpenGL does it! (So actually, OpenGL duplicates the vertices as well, it's not using triangle indices, at least not the way how I draw)
		Triangle tri;
		tri.vertex[0] = mesh.vertex[triIndices[0]];
		tri.vertex[1] = mesh.vertex[triIndices[1]];
		tri.vertex[2] = mesh.vertex[triIndices[2]];
		// 
	}

	//QMatrix4x4 p1;
	//p1.perspective(60, aspect, 1.0f, 100.0f);
	//p1.frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 0.1f, 100.0f);
	//qDebug() << p1;
	
	/*QMatrix4x4 o1;
	o1.ortho(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 0.1f, 100.0f); // l r b t n f
	o1.scale(0.008f);*/
	//o1.scale(0.1f, 0.3f, 1.6f);
	//o1.rotate(389.9f, 0.0f, 0.0f, 1.0f);
	//o1.rotate(197.2f, 0.0f, 1.0f, 0.0f);
	//o1.rotate(45.0f, 1.0f, 0.0f, 0.0f);
	//o1.translate(1.2f, 2.1f, 5.4f);
	
	/*Mat mt1 = utils::MatrixUtils::createTranslationMatrix(1.2f, 2.1f, 5.4f);
	Mat mrx = utils::MatrixUtils::createRotationMatrixX(45.0f * (3.141592f/180.0f));
	Mat mry = utils::MatrixUtils::createRotationMatrixY(197.2f * (3.141592f / 180.0f));
	Mat mrz = utils::MatrixUtils::createRotationMatrixZ(389.9f * (3.141592f / 180.0f));
	Mat ms1 = utils::MatrixUtils::createScalingMatrix(0.1f, 0.3f, 1.6f);
	Mat mp1 = utils::MatrixUtils::createOrthogonalProjectionMatrix(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 0.1f, 100.0f); // l r b t n f
	Mat mpp1 = utils::MatrixUtils::createPerspectiveProjectionMatrix(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 0.1f, 100.0f); // l r b t n f
	Mat mpp2 = utils::MatrixUtils::createPerspectiveProjectionMatrix(60, aspect, 1.0f, 100.0f);*/

	SoftwareRenderer2 swr;
	swr.enableTexturing(true);
	swr.setCurrentTexture(mesh.texture);
	auto swbuffs = swr.render(mesh, matrix);
	Mat swbuffc = swbuffs.first;
	Mat swbuffd = swbuffs.second;
}

} /* namespace render */
