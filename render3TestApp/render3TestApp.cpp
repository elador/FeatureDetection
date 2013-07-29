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
cv::Vec4f matToColVec4f(cv::Mat m) {
	cv::Vec4f ret;
	ret[0] = m.at<float>(0, 0);
	ret[1] = m.at<float>(1, 0);
	ret[2] = m.at<float>(2, 0);
	ret[3] = m.at<float>(3, 0);
	return ret;
}

static Mat colorBuffer;
static Mat depthBuffer;
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

Mat updateProjectionTransform(Camera camera, bool perspective=false) {
	Mat projectionTransform;

	Mat orthogonal = (cv::Mat_<float>(4,4) << 
		2.0f / (camera.frustum.r - camera.frustum.l),	0.0f,											0.0f,											-(camera.frustum.r + camera.frustum.l) / (camera.frustum.r - camera.frustum.l),
		0.0f,											2.0f / (camera.frustum.t - camera.frustum.b),	0.0f,											-(camera.frustum.t + camera.frustum.b) / (camera.frustum.t - camera.frustum.b),
		0.0f,											0.0f,											2.0f / (camera.frustum.n - camera.frustum.f),	-(camera.frustum.n + camera.frustum.f) / (camera.frustum.n - camera.frustum.f), // CG book has denominator (n-f) ? I had (f-n) before. When n and f are neg and here is n-f, then it's the same as n and f pos and f-n here.
		0.0f,											0.0f,											0.0f,											1.0f);
	if (perspective) {
		Mat perspective = (cv::Mat_<float>(4,4) << 
			camera.frustum.n,	0.0f,				0.0f,									0.0f,
			0.0f,				camera.frustum.n,	0.0f,									0.0f,
			0.0f,				0.0f,				camera.frustum.n + camera.frustum.f,	-camera.frustum.n * camera.frustum.f, // CG book has -f*n ? (I had +f*n before). (doesn't matter, cancels when either both n and f neg or both pos)
			0.0f,				0.0f,				+1.0f, /* CG has +1 here, I had -1 */	0.0f);
		projectionTransform = orthogonal * perspective;
	} else {
		projectionTransform = orthogonal;
	}
	return projectionTransform;
}

void renderAxes(Mat worldTransform, Mat viewTransform, Mat projectionTransform, Mat windowTransform) {

	Vec4f origin(0.0f, 0.0f, 0.0f, 1.0f);
	Vec4f xAxis(1.0f, 0.0f, 0.0f, 1.0f);
	Vec4f yAxis(0.0f, 1.0f, 0.0f, 1.0f);
	Vec4f zAxis(0.0f, 0.0f, 1.0f, 1.0f);
	Vec4f mzAxis(0.0f, 0.0f, -1.0f, 1.0f);

	// START Draw the axes:
	Mat worldSpace = worldTransform * Mat(origin);
	Mat camSpace = viewTransform * worldSpace;
	Mat normalizedViewingVolume = projectionTransform * camSpace;
	Vec4f normViewVolVec = matToColVec4f(normalizedViewingVolume);
	normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
	Mat windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
	Vec4f windowCoordsVec = matToColVec4f(windowCoords);
	Point2f originScreen(windowCoordsVec[0], windowCoordsVec[1]);

	worldSpace = worldTransform * Mat(xAxis);
	camSpace = viewTransform * worldSpace;
	normalizedViewingVolume = projectionTransform * camSpace;
	normViewVolVec = matToColVec4f(normalizedViewingVolume);
	normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
	windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
	windowCoordsVec = matToColVec4f(windowCoords);
	Point2f xAxisScreen(windowCoordsVec[0], windowCoordsVec[1]);

	worldSpace = worldTransform * Mat(yAxis);
	camSpace = viewTransform * worldSpace;
	normalizedViewingVolume = projectionTransform * camSpace;
	normViewVolVec = matToColVec4f(normalizedViewingVolume);
	normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
	windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
	windowCoordsVec = matToColVec4f(windowCoords);
	Point2f yAxisScreen(windowCoordsVec[0], windowCoordsVec[1]);

	worldSpace = worldTransform * Mat(zAxis);
	camSpace = viewTransform * worldSpace;
	normalizedViewingVolume = projectionTransform * camSpace;
	normViewVolVec = matToColVec4f(normalizedViewingVolume);
	normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
	windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
	windowCoordsVec = matToColVec4f(windowCoords);
	Point2f zAxisScreen(windowCoordsVec[0], windowCoordsVec[1]);

	worldSpace = worldTransform * Mat(mzAxis);
	camSpace = viewTransform * worldSpace;
	normalizedViewingVolume = projectionTransform * camSpace;
	normViewVolVec = matToColVec4f(normalizedViewingVolume);
	normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
	windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
	windowCoordsVec = matToColVec4f(windowCoords);
	Point2f mzAxisScreen(windowCoordsVec[0], windowCoordsVec[1]);

	line(colorBuffer, originScreen, xAxisScreen, Scalar(0.0f, 0.0f, 255.0f));
	line(colorBuffer, originScreen, yAxisScreen, Scalar(0.0f, 255.0f, 0.0f));
	line(colorBuffer, originScreen, zAxisScreen, Scalar(255.0f, 0.0f, 0.0f));
	line(colorBuffer, originScreen, mzAxisScreen, Scalar(0.0f, 255.0f, 255.0f));
	// END draw axes

}

void renderLine(Vec4f p0, Vec4f p1, Mat worldTransform, Mat viewTransform, Mat projectionTransform, Mat windowTransform) {

	Mat worldSpace = worldTransform * Mat(p0);
	Mat camSpace = viewTransform * worldSpace;
	Mat normalizedViewingVolume = projectionTransform * camSpace;
	Vec4f normViewVolVec = matToColVec4f(normalizedViewingVolume);
	normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
	Mat windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
	Vec4f windowCoordsVec = matToColVec4f(windowCoords);
	Point2f p0Screen(windowCoordsVec[0], windowCoordsVec[1]);

	worldSpace = worldTransform * Mat(p1);
	camSpace = viewTransform * worldSpace;
	normalizedViewingVolume = projectionTransform * camSpace;
	normViewVolVec = matToColVec4f(normalizedViewingVolume);
	normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
	windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
	windowCoordsVec = matToColVec4f(windowCoords);
	Point2f p1Screen(windowCoordsVec[0], windowCoordsVec[1]);

	line(colorBuffer, p0Screen, p1Screen, Scalar(0.0f, 0.0f, 255.0f));
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
	const float aspect = (float)screenWidth/(float)screenHeight;
	Camera camera;
	camera.setFrustum(-1.0f*aspect, 1.0f*aspect, 1.0f, -1.0f, zNear, zFar);
	horizontalAngle = 0.0f; verticalAngle = 0.0f;
	camera.horizontalAngle = horizontalAngle*(CV_PI/180.0f);
	camera.verticalAngle = verticalAngle*(CV_PI/180.0f);
	
	// Cam init: (init + updateFixed)
	Vec3f eye(0.0f, 0.0f, 0.0f);
	//Vec3f gaze(-1.0f, 0.0f, 0.0f);
	Vec3f gaze(0.0f, 0.0f, -1.0f);
	//camera.updateFixed(eye, gaze); // 
	camera.updateFree(eye);  // : given hor/verAngle (and eye), calculate the new FwdVec. Then, new right and up. Then also set at-Vec.

	Mat windowTransform = (cv::Mat_<float>(4,4) << 
		(float)screenWidth/2.0f,		0.0f,						0.0f,	(float)screenWidth/2.0f, // CG book says (screenWidth-1)/2.0f for second value?
		0.0f,							-(float)screenHeight/2.0f,	0.0f,	(float)screenHeight/2.0f,
		0.0f,							0.0f,						1.0f,	0.0f,
		0.0f,							0.0f,						0.0f,	1.0f);

	Mat worldTransform = Mat::eye(4, 4, CV_32FC1);

	Mat translate = render::utils::MatrixUtils::createTranslationMatrix(-camera.getEye()[0], -camera.getEye()[1], -camera.getEye()[2]);
	Mat rotate = (cv::Mat_<float>(4,4) << 
		camera.getRightVector()[0],		camera.getRightVector()[1],		camera.getRightVector()[2],		0.0f,
		camera.getUpVector()[0],		camera.getUpVector()[1],		camera.getUpVector()[2],		0.0f,
		camera.getForwardVector()[0],	camera.getForwardVector()[1],	camera.getForwardVector()[2],	0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);
	Mat viewTransform = rotate * translate;

	Mat projectionTransform = updateProjectionTransform(camera, perspective);

	colorBuffer = Mat::zeros(screenHeight, screenWidth, CV_8UC4);
	depthBuffer = Mat::ones(screenHeight, screenWidth, CV_64FC1)*1000000;
	namedWindow(windowName, WINDOW_AUTOSIZE);
	imshow(windowName, colorBuffer);
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
		

		colorBuffer = Mat::zeros(screenHeight, screenWidth, CV_8UC4);
		depthBuffer = Mat::ones(screenHeight, screenWidth, CV_64FC1)*1000000;
		// Render:
		//camera.update(1):
		// eye += speed * time * forwardVec (or RightVec) (+ or -) = WASD
		// set camera.angles to myangles here:
		camera.horizontalAngle = horizontalAngle*(CV_PI/180.0f);
		camera.verticalAngle = verticalAngle*(CV_PI/180.0f);
		camera.updateFree(camera.eye); // : given hor/verAngle (and eye), calculate the new FwdVec. Then, new right and up. Then also set at-Vec.
		//camera.updateFixed(camera.eye, camera.gaze);
		// update viewTr, projTr
		translate = render::utils::MatrixUtils::createTranslationMatrix(-camera.getEye()[0], -camera.getEye()[1], -camera.getEye()[2]);
		rotate = (cv::Mat_<float>(4,4) << 
			camera.getRightVector()[0],		camera.getRightVector()[1],		camera.getRightVector()[2],		0.0f,
			camera.getUpVector()[0],		camera.getUpVector()[1],		camera.getUpVector()[2],		0.0f,
			camera.getForwardVector()[0],	camera.getForwardVector()[1],	camera.getForwardVector()[2],	0.0f,
			0.0f,				0.0f,				0.0f,				1.0f);
		viewTransform = rotate * translate;

		projectionTransform = updateProjectionTransform(camera, perspective);

		renderAxes(worldTransform, viewTransform, projectionTransform, windowTransform);
		
		for (const auto& tIdx : cube.tvi) {
			Vertex v0 = cube.vertex[tIdx[0]];
			Vertex v1 = cube.vertex[tIdx[1]];
			Vertex v2 = cube.vertex[tIdx[2]];
			renderLine(v0.position, v1.position, worldTransform, viewTransform, projectionTransform, windowTransform);
			renderLine(v1.position, v2.position, worldTransform, viewTransform, projectionTransform, windowTransform);
			renderLine(v2.position, v0.position, worldTransform, viewTransform, projectionTransform, windowTransform);
		}
		
		
		//renderLine(Vec4f(0.0f, 0.0f, -0.1f, 1.0f), Vec4f(0.0f, 0.0f, -100.0f, 1.0f), worldTransform, viewTransform, projectionTransform, windowTransform);
		renderLine(Vec4f(1.5f, 0.0f, 0.5f, 1.0f), Vec4f(-1.5f, 0.0f, 0.5f, 1.0f), worldTransform, viewTransform, projectionTransform, windowTransform);
		

		Mat screenWithOverlay = colorBuffer.clone();
		Mat tmp;
		flip(screenWithOverlay, tmp, 0);
		putText(screenWithOverlay, "(" + lexical_cast<string>(lastX) + ", " + lexical_cast<string>(lastY) + ")", Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screenWithOverlay, "horA: " + lexical_cast<string>(horizontalAngle) + ", verA: " + lexical_cast<string>(verticalAngle), Point(10, 38), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screenWithOverlay, "moving: " + lexical_cast<string>(moving), Point(10, 56), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screenWithOverlay, "zNear: " + lexical_cast<string>(camera.frustum.n) + ", zFar: " + lexical_cast<string>(camera.frustum.f) + ", p: " + lexical_cast<string>(perspective), Point(10, 74), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screenWithOverlay, "eye: " + lexical_cast<string>(camera.eye[0]) + ", " + lexical_cast<string>(camera.eye[1]) + ", " + lexical_cast<string>(camera.eye[2]), Point(10, 92), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		imshow(windowName, screenWithOverlay);

		// n and f have no influence at the moment because I do no clipping?
	}


	
	for (const auto& tIdx : plane.tvi) {
		vector<Vertex> triangle;
		triangle.push_back(plane.vertex[tIdx[0]]);
		triangle.push_back(plane.vertex[tIdx[1]]);
		triangle.push_back(plane.vertex[tIdx[2]]);

		vector<Point> points;
		for (const auto& vtx : triangle) {
			Vec4f vertex = vtx.position;
			// Note: We could put all the points in a matrix and then transform only this matrix?
			Mat worldSpace = worldTransform * Mat(vertex);
			Mat camSpace = viewTransform * worldSpace;
			Mat normalizedViewingVolume = projectionTransform * camSpace;

			// project from 4D to 2D window position with depth value in z coordinate
			Vec4f normViewVolVec = matToColVec4f(normalizedViewingVolume);
			normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
			Mat windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
			Vec4f windowCoordsVec = matToColVec4f(windowCoords);

			//Vec2f p = Vec2f(windowCoordsVec[0], windowCoordsVec[1]);
			points.push_back(Point2f(windowCoordsVec[0], windowCoordsVec[1]));
			
		}/*
		line(colorBuffer, points[0], points[1], Scalar(0.0f, 0.0f, 255.0f));
		line(colorBuffer, points[1], points[2], Scalar(0.0f, 0.0f, 255.0f));
		line(colorBuffer, points[2], points[0], Scalar(0.0f, 0.0f, 255.0f));*/
	}



	
	Vec4f test(0.5f, 0.5f, 0.5f, 1.0f);
	
	return 0;
}

