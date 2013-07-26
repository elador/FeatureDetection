/*
 * OpenGlDevice.cpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */

#include "render/OpenGlDevice.hpp"

#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/highgui/highgui.hpp"

//#include "boost/bind.hpp"

#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>

namespace render {

OpenGlDevice::OpenGlDevice(unsigned int screenWidth, unsigned int screenHeight)
{
	aspect = (float)screenWidth/(float)screenHeight;
	windowName = "OpenGlRenderContext";
	cv::namedWindow(windowName, CV_WINDOW_OPENGL);
	cv::resizeWindow(windowName, screenWidth, screenHeight);
	cv::setOpenGlDrawCallback(windowName, openGlDrawCallback, this);

	//glEnable(GL_TEXTURE_2D);                        // Enable Texture Mapping ( NEW )
	glShadeModel(GL_SMOOTH);                        // Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);                   // Black Background
	glClearDepth(1.0f);                         // Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);                        // Enables Depth Testing
	glDepthFunc(GL_LEQUAL);                         // The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);          // Really Nice Perspective Calculations
}

OpenGlDevice::~OpenGlDevice()
{
	cv::setOpenGlDrawCallback(windowName, 0, 0);
	cv::destroyAllWindows();
}

Mat OpenGlDevice::getImage()
{
	return colorBuffer;
}

Mat OpenGlDevice::getDepthBuffer()
{
	return depthBuffer;
}

// We could try doing that instead with boost::bind, see here: http://stackoverflow.com/questions/282372/demote-boostfunction-to-a-plain-function-pointer
void OpenGlDevice::openGlDrawCallback(void* userdata)
{
	OpenGlDevice* ptr = static_cast<OpenGlDevice*>(userdata);
	ptr->openGlDrawCallbackReal();
}

void OpenGlDevice::openGlDrawCallbackReal()
{
	// init per frame
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);     // Clear The Screen And The Depth Buffer
	glLoadIdentity(); // Reset the modelview Matrix
	// end

	//glViewport ?
	glMatrixMode(GL_PROJECTION);                        // Select The Projection Matrix
	glLoadIdentity();                           // Reset The Projection Matrix
	//glOrtho(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 0.1f, 100.0f);             // Create Ortho View
	//gluPerspective(45, 640.0f/480.0f, 0.1f, 100.0f);
	glFrustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 1.0f, 100.0f); //glFrustum — multiply the current matrix by a perspective matrix

	glMatrixMode(GL_MODELVIEW);                     // Select The Modelview Matrix
	glLoadIdentity();                           // Reset The Modelview Matrix
	// move the camera:
	glTranslatef(0.0f, 0.0f, -1.0f);
	//glTranslatef(0.5f, 0.5f, 0.0f);
	// //gluLookAt(-0.5f, 0.5f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f); ?

	if (!backgroundTex.empty()) {
		glEnable(GL_TEXTURE_2D);
		glColor3f(1.0f, 1.0f, 1.0f);
		backgroundTex.bind();
		glBegin(GL_QUADS);
		glTexCoord2d(0.0, 0.0); glVertex2d(0.0, 0.0);
		glTexCoord2d(1.0, 0.0); glVertex2d(1.0, 0.0);
		glTexCoord2d(1.0, 1.0); glVertex2d(1.0, 1.0);
		glTexCoord2d(0.0, 1.0); glVertex2d(0.0, 1.0);
		glEnd();
		glDisable(GL_TEXTURE_2D);
	}

	drawAxes();
	//drawPlaneXY(1.0f, 1.0f, 1.0f);



}

Vec2f OpenGlDevice::renderVertex(Vec4f vertex)
{
	cv::updateWindow("window"); // When / how often shall I call this? Needed at all?
	return Vec2f();
}

vector<Vec2f> OpenGlDevice::renderVertexList(vector<Vec4f> vertexList)
{
	cv::updateWindow("window");
	return vector<Vec2f>();
}

void OpenGlDevice::setWorldTransform(Mat worldTransform)
{

}

void OpenGlDevice::drawAxes(float scale)
{
	glColor3f(1.0f, 0.0f, 0.0f);	// x-axis
	glBegin(GL_LINES);
		glVertex3f(0.0f, 0.0f, 0.0f);              // From
		glVertex3f(1.0f*scale, 0.0f, 0.0f);              // To
	glEnd();
	glColor3f(0.0f, 1.0f, 0.0f);	// y-axis
	glBegin(GL_LINES);
		glVertex3f(0.0f, 0.0f, 0.0f);              // From
		glVertex3f(0.0f, 1.0f*scale, 0.0f);              // To
	glEnd();
	glColor3f(0.0f, 0.0f, 1.0f);	// z-axis
	glBegin(GL_LINES);
		glVertex3f(0.0f, 0.0f, 0.0f);              // From
		glVertex3f(0.0f, 0.0f, 1.0f*scale);              // To
	glEnd();
	glColor3f(1.0f, 0.0f, 1.0f);	// minus z-axis
	glBegin(GL_LINES);
		glVertex3f(0.0f, 0.0f, 0.0f);              // From
		glVertex3f(0.0f, 0.0f, -1.0f*scale);              // To
	glEnd();
}

void OpenGlDevice::drawBackground()
{

}

void OpenGlDevice::drawPlaneXY(float x, float y, float z, float scale/*=1*/)
{
	glBegin(GL_POLYGON);
	glColor3f( 1.0, 0.0, 0.0 );     glVertex3f(  0.5, -0.5, -0.5 );      // P1 is red
	glColor3f( 1.0, 0.0, 0.0 );     glVertex3f(  0.5,  0.5, -0.5 );      // P2 is green
	glColor3f( 1.0, 0.0, 0.0 );     glVertex3f( -0.5,  0.5, -0.5 );      // P3 is blue
	glColor3f( 1.0, 0.0, 0.0 );     glVertex3f( -0.5, -0.5, -0.5 );      // P4 is purple
	glEnd();
}

void OpenGlDevice::drawCube(float x, float y, float z, float scale/*=1*/)
{
	//Multi-colored side - FRONT
	glBegin(GL_POLYGON);
	glColor3f( 1.0, 0.0, 0.0 );     glVertex3f(  0.5, -0.5, -0.5 );      // P1 is red
	glColor3f( 0.0, 1.0, 0.0 );     glVertex3f(  0.5,  0.5, -0.5 );      // P2 is green
	glColor3f( 0.0, 0.0, 1.0 );     glVertex3f( -0.5,  0.5, -0.5 );      // P3 is blue
	glColor3f( 1.0, 0.0, 1.0 );     glVertex3f( -0.5, -0.5, -0.5 );      // P4 is purple
	glEnd();

	// White side - BACK
	glBegin(GL_POLYGON);
	glColor3f(   1.0,  1.0, 1.0 );
	glVertex3f(  0.5, -0.5, 0.5 );
	glVertex3f(  0.5,  0.5, 0.5 );
	glVertex3f( -0.5,  0.5, 0.5 );
	glVertex3f( -0.5, -0.5, 0.5 );
	glEnd();

	// Purple side - RIGHT
	glBegin(GL_POLYGON);
	glColor3f(  1.0,  0.0,  1.0 );
	glVertex3f( 0.5, -0.5, -0.5 );
	glVertex3f( 0.5,  0.5, -0.5 );
	glVertex3f( 0.5,  0.5,  0.5 );
	glVertex3f( 0.5, -0.5,  0.5 );
	glEnd();

	// Green side - LEFT
	glBegin(GL_POLYGON);
	glColor3f(   0.0,  1.0,  0.0 );
	glVertex3f( -0.5, -0.5,  0.5 );
	glVertex3f( -0.5,  0.5,  0.5 );
	glVertex3f( -0.5,  0.5, -0.5 );
	glVertex3f( -0.5, -0.5, -0.5 );
	glEnd();

	// Blue side - TOP
	glBegin(GL_POLYGON);
	glColor3f(   0.0,  0.0,  1.0 );
	glVertex3f(  0.5,  0.5,  0.5 );
	glVertex3f(  0.5,  0.5, -0.5 );
	glVertex3f( -0.5,  0.5, -0.5 );
	glVertex3f( -0.5,  0.5,  0.5 );
	glEnd();

	// Red side - BOTTOM
	glBegin(GL_POLYGON);
	glColor3f(   1.0,  0.0,  0.0 );
	glVertex3f(  0.5, -0.5, -0.5 );
	glVertex3f(  0.5, -0.5,  0.5 );
	glVertex3f( -0.5, -0.5,  0.5 );
	glVertex3f( -0.5, -0.5, -0.5 );
	glEnd();
}

void OpenGlDevice::setBackgroundImage(Mat background)
{
	backgroundTex.copyFrom(background);
}

void OpenGlDevice::updateWindow()
{
	cv::updateWindow(windowName);
}

} /* namespace render */

/*

GLUquadricObj *quadric;                                     // The Quadric Object
quadric=gluNewQuadric();                                // Create A Pointer To The Quadric Object
gluQuadricNormals(quadric, GLU_SMOOTH);                         // Create Smooth Normals
gluQuadricTexture(quadric, GL_TRUE);                            // Create Texture Coords

gluSphere(quadric,0.1f,32,32);                      // Draw A Sphere
//gluCylinder(quadric,1.5f,1.5f,4.0f,32,16);              // Draw A Cylinder

*/