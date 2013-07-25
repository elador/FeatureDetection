// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#ifdef WIN32
	#include <crtdbg.h>
#endif

#ifdef _DEBUG
   #ifndef DBG_NEW
	  #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
	  #define new DBG_NEW
   #endif
#endif  // _DEBUG

#include "render2/MorphableModel.hpp"
#include "render2/Renderer.hpp"
#include "render2/SoftwareDevice.hpp"
#include "render2/MeshUtils.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/opengl_interop.hpp"
#include <windows.h>
#include <GL/gl.h>
//#include <GL/glu.h>
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"

#include <iostream>
#include <fstream>


namespace po = boost::program_options;
using namespace std;
using namespace cv;



void on_opengl(void* userdata)
{
	cv::GlTexture2D* pTex = static_cast<cv::GlTexture2D*>(userdata);
	if (pTex->empty())
		return;

	// INIT (once)
	glEnable(GL_TEXTURE_2D);                        // Enable Texture Mapping ( NEW )
	glShadeModel(GL_SMOOTH);                        // Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);                   // Black Background
	glClearDepth(1.0f);                         // Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);                        // Enables Depth Testing
	glDepthFunc(GL_LEQUAL);                         // The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);          // Really Nice Perspective Calculations
	// INIT END
	
	// init per frame
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);     // Clear The Screen And The Depth Buffer
	glLoadIdentity();
	// end

	//glScalef(0.1f, 0.1f, 0.1f);
	/*
	glTranslatef(-1.5f,0.0f, 0.0f);                 // Move Left 1.5 Units And Into The Screen 6.0
	glBegin(GL_TRIANGLES);                      // Drawing Using Triangles
		glColor3f(1.0f,0.0f,0.0f);          // Set The Color To Red
		glVertex3f( 0.0f, 1.0f, 0.0f);              // Top
		glColor3f(0.0f,1.0f,0.0f);          // Set The Color To Green
		glVertex3f(-1.0f,-1.0f, 0.0f);              // Bottom Left
		glColor3f(0.0f,0.0f,1.0f);          // Set The Color To Blue
		glVertex3f( 1.0f,-1.0f, 0.0f);              // Bottom Right
	glEnd();                            // Finished Drawing The Triangle

	glTranslatef(3.0f,0.0f,0.0f);                   // Move Right 3 Units
	glColor3f(0.5f,0.5f,1.0f);              // Set The Color To Blue One Time Only
	glBegin(GL_QUADS);                      // Draw A Quad
		glVertex3f(-1.0f, 1.0f, 0.0f);              // Top Left
		glVertex3f( 1.0f, 1.0f, 0.0f);              // Top Right
		glVertex3f( 1.0f,-1.0f, 0.0f);              // Bottom Right
		glVertex3f(-1.0f,-1.0f, 0.0f);              // Bottom Left
	glEnd();                            // Done Drawing The Quad
	*/
	glColor3f(1.0f, 0.0f, 0.0f);	// x-axis
	glBegin(GL_LINES);
		glVertex3f(0.0f, 0.0f, 0.0f);              // From
		glVertex3f( 1.0f, 0.0f, 0.0f);              // To
	glEnd();
	glColor3f(0.0f, 1.0f, 0.0f);	// y-axis
	glBegin(GL_LINES);
		glVertex3f(0.0f, 0.0f, 0.0f);              // From
		glVertex3f( 0.0f, 1.0f, 0.0f);              // To
	glEnd();
	glColor3f(0.0f, 0.0f, 1.0f);	// z-axis
	glBegin(GL_LINES);
		glVertex3f(0.0f, 0.0f, 0.0f);              // From
		glVertex3f( 0.0f, 0.0f, 1.0f);              // To
	glEnd();


	/*
	glEnable(GL_TEXTURE_2D);
	pTex->bind();
	glBegin( GL_QUADS );
		glTexCoord2d(0.0,0.0); glVertex2d(0.0,0.0);
		glTexCoord2d(1.0,0.0); glVertex2d(1.0,0.0);
		glTexCoord2d(1.0,1.0); glVertex2d(1.0,1.0);
		glTexCoord2d(0.0,1.0); glVertex2d(0.0,1.0);
	glEnd();
	*/

}


template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
	return os;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(3759128);
	#endif

	std::string filename; // Create vector to hold the filenames
	
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("input-file,i", po::value<string>(), "input image")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).
				  options(desc).run(), vm);
		po::notify(vm);
	
		if (vm.count("help")) {
			cout << "[renderTestApp] Usage: options_description [options]\n";
			cout << desc;
			return 0;
		}
		if (vm.count("input-file"))
		{
			cout << "[renderTestApp] Using input images: " << vm["input-file"].as< vector<string> >() << "\n";
			filename = vm["input-file"].as<string>();
		}
	}
	catch(std::exception& e) {
		cout << e.what() << "\n";
		return 1;
	}


	//render::MorphableModel mmHeadL4 = render::utils::MeshUtils::readFromScm("C:\\Users\\Patrik\\Cloud\\PhD\\MorphModel\\ShpVtxModelBin.scm");

	/*
	const string windowName = "OpenGL Sample";
	cv::namedWindow(windowName, cv::WINDOW_OPENGL);
	cv::resizeWindow(windowName, 640, 480);
	cv::setOpenGlDrawCallback(windowName, on_opengl);*/
	//cv::setOpenGlContext(windowName);

	const float& aspect = 640.0f/480.0f;
	shared_ptr<render::RenderDevice> swr = make_shared<render::SoftwareDevice>(640, 480);
	render::Renderer r(swr);
	r.renderDevice->getCamera().setFrustum(-1.0f*aspect, 1.0f*aspect, 1.0f, -1.0f, 0.1f, 100.0f);
	Vec4f test(0.5f, 0.5f, 0.5f, 1.0f);
	r.renderDevice->renderVertex(test);



#pragma comment(lib, "opengl32")
//#pragma comment(lib, "glu32")
	
	cv::VideoCapture webcam(CV_CAP_ANY);

	cv::namedWindow("window", CV_WINDOW_OPENGL);
	cv::resizeWindow("window", 640, 480);

	Mat frame;
	//cv::Mat frame = imread("C:/Users/Patrik/Cloud/may_prag_vs_basel.png");
	//resize(frame, frame, Size(640, 480));
	cv::GlTexture2D tex;

	cv::setOpenGlDrawCallback("window", on_opengl, &tex);

	while(cv::waitKey(30) < 0)
	{
		webcam >> frame;
		tex.copyFrom(frame);
		cv::updateWindow("window");
	}

	cv::setOpenGlDrawCallback("window", 0, 0);
	cv::destroyAllWindows();

	


	
	return 0;
}

// TODO: All those vec3 * mat4 cases... Do I have to add the homogeneous coordinate and make it vec4 * mat4, instead of how I'm doing it now? Difference?