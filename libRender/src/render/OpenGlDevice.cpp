/*
 * OpenGlDevice.cpp
 *
 *  Created on: 23.07.2013
 *      Author: Patrik Huber
 */

#include "render/OpenGlDevice.hpp"

#include "opencv2/core/core.hpp" // check if needed
#include "opencv2/imgproc/imgproc.hpp" // check if needed
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/highgui/highgui.hpp"

//#include "boost/bind.hpp"

#include <windows.h> // #idef WIN32....
#include <GL/gl.h>
#include <GL/glu.h>

#include <iostream>

namespace render {

/* What follows is OGL_OCV_Common */

void copyImgToTex(const Mat& _tex_img, GLuint* texID, double* _twr, double* _thr);

typedef struct my_texture {
	GLuint tex_id;
	double twr,thr,aspect_w2h;
	Mat image;
	my_texture():tex_id(-1),twr(1.0),thr(1.0) {}
	bool initialized;
	void set(const Mat& ocvimg) { 
		ocvimg.copyTo(image); 
		copyImgToTex(image, &tex_id, &twr, &thr); 
		aspect_w2h = (double)ocvimg.cols/(double)ocvimg.rows;
	}
} OpenCVGLTexture;

void glEnable2D();	// setup 2D drawing
void glDisable2D(); // end 2D drawing
OpenCVGLTexture MakeOpenCVGLTexture(const Mat& _tex_img); // create an OpenCV-OpenGL image
void drawOpenCVImageInGL(const OpenCVGLTexture& tex); // draw an OpenCV-OpenGL image

void drawOpenCVImageInGL(const OpenCVGLTexture& tex) {
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);

	int vPort[4]; glGetIntegerv(GL_VIEWPORT, vPort);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, tex.tex_id);
	glPushMatrix();
	glColor3ub(255, 255, 255);

	glScaled(vPort[3], vPort[3], 1);

	double aw2h = tex.aspect_w2h, ithr = tex.thr, itwr = tex.twr;
	double n[3] = {0,0,-1};

	GLint face_ori; glGetIntegerv(GL_FRONT_FACE, &face_ori);
	glFrontFace(GL_CW); //we're gonna draw clockwise

	glBegin(GL_QUADS);	

	glTexCoord2d(0, 0);
	glNormal3dv(n); 
	glVertex2d(0, 0);

	glTexCoord2d(0, ithr);
	glNormal3dv(n); 
	glVertex2d(0, 1); 

	glTexCoord2d(itwr, ithr);	
	glNormal3dv(n); 
	glVertex2d(aw2h, 1);

	glTexCoord2d(itwr, 0);				
	glNormal3dv(n); 
	glVertex2d(aw2h, 0); 

	glEnd();
	glPopMatrix();

	glFrontFace(face_ori); //restore front face orientation

	glDisable(GL_TEXTURE_2D);
	glEnable(GL_LIGHTING);
	glEnable(GL_BLEND);
}

void glEnable2D()
{
	int vPort[4];

	glGetIntegerv(GL_VIEWPORT, vPort);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glOrtho(0, vPort[2], 0, vPort[3], -1, 4);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glTranslated(0.375, 0.375, 0);

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT); // clear the screen

	glDisable(GL_DEPTH_TEST);	
}

void glDisable2D()
{
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();   
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();	

	glEnable(GL_DEPTH_TEST);
}

#if defined(WIN32)
#define log2(x) log10((double)(x))/log10(2.0)
#endif

void copyImgToTex(const Mat& _tex_img, GLuint* texID, double* _twr, double* _thr) {
	Mat tex_img = _tex_img;
	flip(_tex_img,tex_img,0);
	Mat tex_pow2(pow(2.0,ceil(log2(tex_img.rows))),pow(2.0,ceil(log2(tex_img.cols))),CV_8UC3);
	std::cout << tex_pow2.rows <<"x"<<tex_pow2.cols<<std::endl;
	Mat region = tex_pow2(cv::Rect(0,0,tex_img.cols,tex_img.rows));
	if (tex_img.type() == region.type()) {
		tex_img.copyTo(region);
	} else if (tex_img.type() == CV_8UC1) {
		cvtColor(tex_img, region, CV_GRAY2BGR);
	} else {
		tex_img.convertTo(region, CV_8UC3, 255.0);
	}

	if (_twr != 0 && _thr != 0) {
		*_twr = (double)tex_img.cols/(double)tex_pow2.cols;
		*_thr = (double)tex_img.rows/(double)tex_pow2.rows;
	}
	glBindTexture( GL_TEXTURE_2D, *texID );
	glTexImage2D(GL_TEXTURE_2D, 0, 3, tex_pow2.cols, tex_pow2.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, tex_pow2.data);
}	


OpenCVGLTexture MakeOpenCVGLTexture(const Mat& _tex_img) {
	OpenCVGLTexture _ocvgl;

	glGenTextures( 1, &_ocvgl.tex_id );
	glBindTexture( GL_TEXTURE_2D, _ocvgl.tex_id );
	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

	if (!_tex_img.empty()) { //image may be dummy, just to generate pointer to texture object in GL
		copyImgToTex(_tex_img,&_ocvgl.tex_id,&_ocvgl.twr,&_ocvgl.thr);
		_ocvgl.aspect_w2h = (double)_tex_img.cols/(double)_tex_img.rows;
	}

	return _ocvgl;
}

// END OGL_OCV_Common


OpenGlDevice::OpenGlDevice(unsigned int screenWidth, unsigned int screenHeight)
{
	//renderTypeState = RenderTypeState::NONE;
	renderTypeState = RenderTypeState::MESH;
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

	// From PnP example:
/*	tv = vector<double>(3);

	const GLfloat light_ambient[]  = { 0.0f, 0.0f, 0.0f, 1.0f };
	const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
	const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	const GLfloat light_position[] = { 0.0f, 0.0f, 1.0f, 0.0f };

	const GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
	const GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
	const GLfloat mat_specular[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
	const GLfloat high_shininess[] = { 100.0f };
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);


	glShadeModel(GL_SMOOTH);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial ( GL_FRONT, GL_AMBIENT_AND_DIFFUSE );

	glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

	glEnable(GL_LIGHTING);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();*/
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
	// if (ptr)
	ptr->openGlDrawCallbackInstance();
}

// rename to draw()?
void OpenGlDevice::openGlDrawCallbackInstance()
{
	// init per frame
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);     // Clear The Screen And The Depth Buffer
	glLoadIdentity(); // Reset the modelview Matrix
	// end

	//glViewport ?
	glMatrixMode(GL_PROJECTION);                        // Select The Projection Matrix
	glLoadIdentity();                           // Reset The Projection Matrix
	glOrtho(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 0.1f, 100.0f);             // Create Ortho View
	//gluPerspective(45, 640.0f/480.0f, 0.1f, 100.0f);
	//glFrustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 1.0f, 100.0f); //glFrustum — multiply the current matrix by a perspective matrix

	glMatrixMode(GL_MODELVIEW);                     // Select The Modelview Matrix
	glLoadIdentity();                           // Reset The Modelview Matrix
	// move the camera:
	
	//glRotatef( 20.0f, 1.0f, 0.0f, 0.0f);
	//glRotatef( 20.0f, 0.0f, 1.0f, 0.0f);
	//glTranslatef(0.0f, 0.0f, -1.5f);
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

	
	glTranslated(t.at<float>(0), t.at<float>(1), t.at<float>(2));
	double _d[16] = {	r.at<float>(0), r.at<float>(1),r.at<float>(2), 0,
		r.at<float>(0), r.at<float>(4), r.at<float>(5), 0,
		r.at<float>(6), r.at<float>(7), r.at<float>(8), 0,
		0,	   0,	  0		,1};
	glMultMatrixd(_d);

	//glScalef(1.0f/90.0f, 1.0f/90.0f, 1.0f/90.0f);
	if (renderTypeState==RenderTypeState::MESH) {
		drawMesh();
	}

	//updateWindow();

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
	glColor3f(0.0f, 0.0f, 1.0f);	// z-axis (coming to front)
	glBegin(GL_LINES);
		glVertex3f(0.0f, 0.0f, 0.0f);              // From
		glVertex3f(0.0f, 0.0f, 1.0f*scale);              // To
	glEnd();
	glColor3f(1.0f, 0.0f, 1.0f);	// minus z-axis (violet)
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

void OpenGlDevice::renderMesh(Mesh mesh)
{
	meshToDraw = mesh;
	renderTypeState = RenderTypeState::MESH;
}

void OpenGlDevice::drawMesh()
{
	for (const auto& tIdx : meshToDraw.tvi) {
		Vertex v0 = meshToDraw.vertex[tIdx[0]];
		Vertex v1 = meshToDraw.vertex[tIdx[1]];
		Vertex v2 = meshToDraw.vertex[tIdx[2]];
		glBegin(GL_TRIANGLES);
			glColor3f(v0.color[0],  v0.color[1],  v0.color[2]); // BGR to RGB
			glVertex3f(v0.position[0], v0.position[1], v0.position[2]);
			glColor3f(v1.color[0],  v1.color[1],  v1.color[2]); // BGR to RGB
			glVertex3f(v1.position[0], v1.position[1], v1.position[2]);	
			glColor3f(v2.color[0],  v2.color[1],  v2.color[2]); // BGR to RGB
			glVertex3f(v2.position[0], v2.position[1], v2.position[2]);
		glEnd();
	}

}

void OpenGlDevice::drawVertex(Vertex v)
{

}

void OpenGlDevice::saveOpenGLBuffer()
{
	static unsigned int opengl_buffer_num = 0;

	int vPort[4]; glGetIntegerv(GL_VIEWPORT, vPort);
	cv::Mat_<cv::Vec3b> opengl_image(vPort[3],vPort[2]);
	{
		cv::Mat_<cv::Vec4b> opengl_image_4b(vPort[3],vPort[2]);
		glReadPixels(0, 0, vPort[2], vPort[3], GL_BGRA_EXT, GL_UNSIGNED_BYTE, opengl_image_4b.data);
		cv::flip(opengl_image_4b,opengl_image_4b,0);
		mixChannels(&opengl_image_4b, 1, &opengl_image, 1, &(cv::Vec6i(0,0,1,1,2,2)[0]), 3);
	}
	std::stringstream ss; ss << "opengl_buffer_" << opengl_buffer_num++ << ".jpg";
	imwrite(ss.str(), opengl_image);

}

void OpenGlDevice::resize(int width, int height)
{

	const float ar = (float) width / (float) height;

	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//glFrustum(-ar, ar, -1.0, 1.0, 2.0, 100.0);
	gluPerspective(47,1.0,0.01,1000.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void OpenGlDevice::drawAxesNew() {

	//Z = red
	glPushMatrix();
	glRotated(180,0,1,0);
	glColor4d(1,0,0,0.5);
	//	glutSolidCylinder(0.05,1,15,20);
	glBegin(GL_LINES);
	glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
	glEnd();
	glTranslated(0,0,1);
	glScaled(.1,.1,.1);
	//glutSolidTetrahedron();
	glPopMatrix();

	//Y = green
	glPushMatrix();
	glRotated(-90,1,0,0);
	glColor4d(0,1,0,0.5);
	//	glutSolidCylinder(0.05,1,15,20);
	glBegin(GL_LINES);
	glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
	glEnd();
	glTranslated(0,0,1);
	glScaled(.1,.1,.1);
	//glutSolidTetrahedron();
	glPopMatrix();

	//X = blue
	glPushMatrix();
	glRotated(-90,0,1,0);
	glColor4d(0,0,1,0.5);
	//	glutSolidCylinder(0.05,1,15,20);
	glBegin(GL_LINES);
	glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
	glEnd();
	glTranslated(0,0,1);
	glScaled(.1,.1,.1);
	//glutSolidTetrahedron();
	glPopMatrix();
}

void OpenGlDevice::display()
{	
	// draw the image in the back
/*	int vPort[4]; glGetIntegerv(GL_VIEWPORT, vPort);
	glEnable2D();
	drawOpenCVImageInGL(imgTex);
	glTranslated(vPort[2]/2.0, 0, 0);
	drawOpenCVImageInGL(imgWithDrawing);
	glDisable2D();

	glClear(GL_DEPTH_BUFFER_BIT); // we want to draw stuff over the image

	// draw only on left part
	glViewport(0, 0, vPort[2]/2, vPort[3]);

	glPushMatrix();

	gluLookAt(0,0,0,0,0,1,0,-1,0);

	// put the object in the right position in space
	cv::Vec3d tvv(tv[0],tv[1],tv[2]);
	glTranslated(tvv[0], tvv[1], tvv[2]);

	// rotate it
	double _d[16] = {	rot[0],rot[1],rot[2],0,
		rot[3],rot[4],rot[5],0,
		rot[6],rot[7],rot[8],0,
		0,	   0,	  0		,1};
	glMultMatrixd(_d);

	// draw the 3D head model
	glColor4f(1, 1, 1,0.75);
	glmDraw(head_obj, GLM_SMOOTH);

	//----------Axes
	glScaled(50, 50, 50);
	drawAxes();
	//----------End axes

	glPopMatrix();

	// restore to looking at complete viewport
	glViewport(0, 0, vPort[2], vPort[3]); 

	glutSwapBuffers();*/
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