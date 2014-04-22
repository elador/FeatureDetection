/*
 * QOpenGLRenderer.hpp
 *
 *  Created on: 04.04.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef QOPENGLRENDERER_HPP_
#define QOPENGLRENDERER_HPP_

#ifdef WITH_RENDER_QOPENGL

#include "QtGui/QOpenGLFunctions"
#include "QtGui/QOpenGLShaderProgram"

namespace render {

class Mesh;

/**
 * Desc
 */
class QOpenGLRenderer : protected QOpenGLFunctions
{
public:
	QOpenGLRenderer(QOpenGLContext* qOpenGlContext); // maybe change to QObject (QObject*). Should work as long as the QObject is a window that contains a OpenGL surface/context.
	// Todo: should go to c'tor?
	void setViewport(int viewportWidth, int viewportHeight, qreal retinaScale) {
		this->viewportWidth = viewportWidth;
		this->viewportHeight = viewportHeight;
		this->retinaScale = retinaScale;
	};
	// maybe make a setMesh(), to avoid copying the mesh in every frame
	void render(Mesh mesh, QMatrix4x4 mvp);

	// Warning: Performs full rendering of the frame in the current OpenGL context.
	// getFramebuffer(Mesh, QMatrix4x4);
private:
	QOpenGLShaderProgram *m_program;
	QOpenGLContext* qOpenGlContext;

	GLuint m_posAttr;
	GLuint m_colAttr;
	GLuint m_texAttr;
	GLuint m_texWeightAttr;
	GLuint m_matrixUniform;

	GLuint texture;

	int m_frame;

	int viewportWidth;
	int viewportHeight;
	qreal retinaScale;
};

static const char *vertexShaderSource =
"attribute highp vec4 posAttr;\n"
"attribute lowp vec4 colAttr;\n"
"attribute mediump vec2 texAttr;\n" // new tex
"attribute mediump float texWeightAttr;\n" // new tex
"varying lowp vec4 col;\n"
"varying mediump vec2 tex;\n" // new tex
"uniform highp mat4 matrix;\n"
"varying lowp float texWeight;\n" // new tex
"void main() {\n"
"   col = colAttr;\n"
"   tex = texAttr;\n" // new tex
"   texWeight = texWeightAttr;\n" // new tex
"   gl_Position = matrix * posAttr;\n"
"}\n";

static const char *fragmentShaderSource =
"varying lowp vec4 col;\n"
"varying mediump vec2 tex;\n" // new tex
"varying lowp float texWeight;\n" // new tex
"uniform sampler2D texture;\n" // new tex
"void main() {\n"
//    "   gl_FragColor = col;\n"
//	"   gl_FragColor = texture2D(texture, tex);\n" // new tex
"   gl_FragColor = mix(col, texture2D(texture, tex), texWeight);\n" // new tex
//"   gl_FragColor = vec4(0.0, texWeight, 0.0, 0.0);\n"
//"   gl_FragColor = vec4(tex.x, tex.y, 0.0, 0.0);\n"
"}\n";

 } /* namespace render */

#endif /* WITH_RENDER_QOPENGL */

#endif /* QOPENGLRENDERER_HPP_ */
