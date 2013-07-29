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

#include "render/MorphableModel.hpp"
#include "render/MeshUtils.hpp"
#include "render/MatrixUtils.hpp"
#include "render/RenderDevice.hpp"
#include "render/Camera.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/lexical_cast.hpp"

#include <iostream>
#include <fstream>

namespace po = boost::program_options;
using namespace std;
using namespace cv;
using namespace render;
using boost::lexical_cast;

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
	return os;
}

static string windowName = "test";
static float horizontalAngle = 0.0f;
static float verticalAngle = 0.0f;
static int lastX = 0;
static int lastY = 0;
static float movingFactor = 0.5f;
static bool moving = false;

static float zNear = -0.1f;
static float zFar = -100.0f;
static bool perspective = false;

static void winOnMouse(int event, int x, int y, int, void* userdata)
{
	if (event == EVENT_MOUSEMOVE && moving == false) {
		lastX = x;
		lastY = y;
	}
	if (event == EVENT_MOUSEMOVE && moving == true) {
		if (x != lastX || y != lastY) {
			horizontalAngle += (x-lastX)*movingFactor;
			verticalAngle += (y-lastY)*movingFactor;
			lastX = x;
			lastY = y;
		}

		//imshow(windowName, colorBuffer);
	}
	if (event == EVENT_LBUTTONDOWN) {
		moving = true;
		lastX = x;
		lastY = y;
	}
	if (event == EVENT_LBUTTONUP) {
		moving = false;
	}

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
			cout << "[renderTestApp] Using input images: " << vm["input-file"].as<vector<string>>() << "\n";
			filename = vm["input-file"].as<string>();
		}
	}
	catch(std::exception& e) {
		cout << e.what() << "\n";
		return 1;
	}
	
	//render::MorphableModel mmHeadL4 = render::utils::MeshUtils::readFromScm("C:\\Users\\Patrik\\Cloud\\PhD\\MorphModel\\ShpVtxModelBin.scm");
	render::Mesh cube = render::utils::MeshUtils::createCube();
	render::Mesh pyramid = render::utils::MeshUtils::createPyramid();
	render::Mesh plane = render::utils::MeshUtils::createPlane();

	int screenWidth = 640;
	int screenHeight = 480;
	RenderDevice r(screenWidth, screenHeight);
	const float aspect = (float)screenWidth/(float)screenHeight;
	Camera camera;
	camera.setFrustum(-1.0f*aspect, 1.0f*aspect, 1.0f, -1.0f, zNear, zFar);
	r.camera = camera;

	horizontalAngle = 0.0f; verticalAngle = 0.0f;
	camera.horizontalAngle = horizontalAngle*(CV_PI/180.0f);
	camera.verticalAngle = verticalAngle*(CV_PI/180.0f);
	
	// Cam init: (init + updateFixed)
	Vec3f eye(0.0f, 0.0f, 0.0f);
	//Vec3f gaze(-1.0f, 0.0f, 0.0f);
	Vec3f gaze(0.0f, 0.0f, -1.0f);
	//camera.updateFixed(eye, gaze); // 
	camera.updateFree(eye);  // : given hor/verAngle (and eye), calculate the new FwdVec. Then, new right and up. Then also set at-Vec.

	namedWindow(windowName, WINDOW_AUTOSIZE);
	setMouseCallback(windowName, winOnMouse);

	bool running = true;
	while (running) {
		int key = waitKey(30);
		if (key==-1) {
			// no key pressed
		}
		if (key == 'q') {
			running = false;
		}
		if (key == 'w') {
			camera.eye += 0.05f * -camera.getForwardVector();
		}
		if (key == 'a') {
			camera.eye -= 0.05f * camera.getRightVector();
		}
		if (key == 's') {
			camera.eye -= 0.05f * -camera.getForwardVector();
		}
		if (key == 'd') {
			camera.eye += 0.05f * camera.getRightVector();
		}
		if (key == 'r') {
			camera.eye += 0.05f * camera.getUpVector();
		}
		if (key == 'f') {
			camera.eye -= 0.05f * camera.getUpVector();
		}

		if (key == 'z') {
			camera.gaze += 0.05f * camera.getUpVector();
		}
		if (key == 'h') {
			camera.gaze -= 0.05f * camera.getUpVector();
		}
		if (key == 'g') {
			camera.gaze -= 0.05f * camera.getRightVector();
		}
		if (key == 'j') {
			camera.gaze += 0.05f * camera.getRightVector();
		}

		if (key == 'i') {
			camera.frustum.n += 0.1f;
		}
		if (key == 'k') {
			camera.frustum.n -= 0.1f;
		}
		if (key == 'o') {
			camera.frustum.f += 1.0f;
		}
		if (key == 'l') {
			camera.frustum.f -= 1.0f;
		}
		if (key == 'p') {
			perspective = !perspective;
		}
		
		r.resetBuffers();

		// Render:
		//camera.update(1):
		// eye += speed * time * forwardVec (or RightVec) (+ or -) = WASD
		// set camera.angles to myangles here:
		camera.horizontalAngle = horizontalAngle*(CV_PI/180.0f);
		camera.verticalAngle = verticalAngle*(CV_PI/180.0f);
		camera.updateFree(camera.eye); // : given hor/verAngle (and eye), calculate the new FwdVec. Then, new right and up. Then also set at-Vec.
		//camera.updateFixed(camera.eye, camera.gaze);
		r.camera = camera;
		// update viewTr, projTr
		r.updateViewTransform();
		r.updateProjectionTransform(false);

		Vec4f origin(0.0f, 0.0f, 0.0f, 1.0f);
		Vec4f xAxis(1.0f, 0.0f, 0.0f, 1.0f);
		Vec4f yAxis(0.0f, 1.0f, 0.0f, 1.0f);
		Vec4f zAxis(0.0f, 0.0f, 1.0f, 1.0f);
		Vec4f mzAxis(0.0f, 0.0f, -1.0f, 1.0f);
		r.renderLine(origin, xAxis, Scalar(0.0f, 0.0f, 255.0f));
		r.renderLine(origin, yAxis, Scalar(0.0f, 255.0f, 0.0f));
		r.renderLine(origin, zAxis, Scalar(255.0f, 0.0f, 0.0f));
		r.renderLine(origin, mzAxis, Scalar(0.0f, 255.0f, 255.0f));
		
		for (const auto& tIdx : cube.tvi) {
			Vertex v0 = cube.vertex[tIdx[0]];
			Vertex v1 = cube.vertex[tIdx[1]];
			Vertex v2 = cube.vertex[tIdx[2]];
			r.renderLine(v0.position, v1.position, Scalar(0.0f, 0.0f, 255.0f));
			r.renderLine(v1.position, v2.position, Scalar(0.0f, 0.0f, 255.0f));
			r.renderLine(v2.position, v0.position, Scalar(0.0f, 0.0f, 255.0f));
		}
		
		r.renderLine(Vec4f(1.5f, 0.0f, 0.5f, 1.0f), Vec4f(-1.5f, 0.0f, 0.5f, 1.0f), Scalar(0.0f, 0.0f, 255.0f));

		Mat screen = r.getImage();
		putText(screen, "(" + lexical_cast<string>(lastX) + ", " + lexical_cast<string>(lastY) + ")", Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "horA: " + lexical_cast<string>(horizontalAngle) + ", verA: " + lexical_cast<string>(verticalAngle), Point(10, 38), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "moving: " + lexical_cast<string>(moving), Point(10, 56), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "zNear: " + lexical_cast<string>(camera.frustum.n) + ", zFar: " + lexical_cast<string>(camera.frustum.f) + ", p: " + lexical_cast<string>(perspective), Point(10, 74), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "eye: " + lexical_cast<string>(camera.eye[0]) + ", " + lexical_cast<string>(camera.eye[1]) + ", " + lexical_cast<string>(camera.eye[2]), Point(10, 92), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		imshow(windowName, screen);

		// n and f have no influence at the moment because I do no clipping?
	}
	
	return 0;
}

