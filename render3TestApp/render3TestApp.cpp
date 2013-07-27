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
static float movingFactor = 1.0f;
static bool moving = false;

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
	const float aspect = (float)screenWidth/(float)screenHeight;
	bool perspective = false;
	Camera camera;
	camera.setFrustum(-1.0f*aspect, 1.0f*aspect, 1.0f, -1.0f, 0.1f, 100.0f);
	horizontalAngle = 0.0f; verticalAngle = 0.0f;
	// Cam init: (init + updateFixed)
	Vec3f eye(0.0f, 0.0f, 3.0f);
	Vec3f at(0.0f, 0.0f, 0.0f);
	camera.updateFixed(eye, at); // is fwdVec now 0, 0, -1 ?

	Mat windowTransform = (cv::Mat_<float>(4,4) << 
		(float)screenWidth/2.0f,		0.0f,						0.0f,	(float)screenWidth/2.0f, // CG book says (screenWidth-1)/2.0f for second value?
		0.0f,							(float)screenHeight/2.0f,	0.0f,	(float)screenHeight/2.0f,
		0.0f,							0.0f,						1.0f,	0.0f,
		0.0f,							0.0f,						0.0f,	1.0f);

	Mat worldTransform = Mat::eye(4, 4, CV_32FC1);

	Mat translate = render::utils::MatrixUtils::createTranslationMatrix(-camera.getEye()[0], -camera.getEye()[1], -camera.getEye()[2]);
	Mat rotate = (cv::Mat_<float>(4,4) << 
		camera.getRightVector()[0],		camera.getRightVector()[1],		camera.getRightVector()[2],		0.0f,
		camera.getUpVector()[0],		camera.getUpVector()[1],		camera.getUpVector()[2],		0.0f,
		-camera.getForwardVector()[0],	-camera.getForwardVector()[1],	-camera.getForwardVector()[2],	0.0f,
		0.0f,				0.0f,				0.0f,				1.0f);
	Mat viewTransform = rotate * translate;

	Mat projectionTransform;
	Mat orthogonal = (cv::Mat_<float>(4,4) << 
		2.0f / (camera.frustum.r - camera.frustum.l),	0.0f,											0.0f,											-(camera.frustum.r + camera.frustum.l) / (camera.frustum.r - camera.frustum.l),
		0.0f,											2.0f / (camera.frustum.t - camera.frustum.b),	0.0f,											-(camera.frustum.t + camera.frustum.b) / (camera.frustum.t - camera.frustum.b),
		0.0f,											0.0f,											2.0f / (camera.frustum.n - camera.frustum.f),	-(camera.frustum.n + camera.frustum.f) / (camera.frustum.f - camera.frustum.n), // CG book has denominator (n-f) ? I had (f-n) before.
		0.0f,											0.0f,											0.0f,											1.0f);
	if (perspective) {
		Mat perspective = (cv::Mat_<float>(4,4) << 
			camera.frustum.n,	0.0f,				0.0f,									0.0f,
			0.0f,				camera.frustum.n,	0.0f,									0.0f,
			0.0f,				0.0f,				camera.frustum.n + camera.frustum.f,	+camera.frustum.n * camera.frustum.f, // CG book has -f*n ? (I had +f*n before)
			0.0f,				0.0f,				-1.0f, /* CG has +1 here, I had -1 */	0.0f);
		projectionTransform = orthogonal * perspective;
	} else {
		projectionTransform = orthogonal;
	}


	colorBuffer = Mat::zeros(screenHeight, screenWidth, CV_8UC4);
	depthBuffer = Mat::ones(screenHeight, screenWidth, CV_64FC1)*1000000;
	namedWindow(windowName, WINDOW_AUTOSIZE);
	imshow(windowName, colorBuffer);
	setMouseCallback(windowName, winOnMouse);

	// Render:
	//camera.update(1):
	// eye += speed * time * forwardVec (or RightVec) (+ or -) = WASD
	camera.updateFree(eye); // : given hor/verAngle (and eye), calculate the new FwdVec. Then, new right and up. Then also set at-Vec.
	// update viewTr, projTr
	
	vector<Vec2f> pointList;
	for (const auto& vertex : vertexList) {		// Note: We could put all the points in a matrix and then transform only this matrix?
		Mat worldSpace = worldTransform * Mat(vertex);
		Mat camSpace = viewTransform * worldSpace;
		Mat normalizedViewingVolume = projectionTransform * camSpace;

		// project from 4D to 2D window position with depth value in z coordinate
		Vec4f normViewVolVec = matToColVec4f(normalizedViewingVolume);
		normViewVolVec = normViewVolVec / normViewVolVec[3];	// divide by w
		Mat windowCoords = windowTransform * Mat(normViewVolVec);	// places the vec as a column in the matrix
		Vec4f windowCoordsVec = matToColVec4f(windowCoords);

		Vec2f p = Vec2f(windowCoordsVec[0], windowCoordsVec[1]);
		pointList.push_back(p);
	}


	bool running = true;
	while (running) {
		int key = waitKey(30);
		if (key==-1) {
			// no key pressed
		}
		if (key == 'q') {
			running = false;
		}
		Mat screenWithOverlay = colorBuffer.clone();
		putText(screenWithOverlay, "(" + lexical_cast<string>(lastX) + ", " + lexical_cast<string>(lastY) + ")", Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 0, 255));
		putText(screenWithOverlay, "horA: " + lexical_cast<string>(horizontalAngle) + ", verA: " + lexical_cast<string>(verticalAngle), Point(10, 45), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 0, 255));
		putText(screenWithOverlay, "moving: " + lexical_cast<string>(moving), Point(10, 70), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 0, 255));
		imshow(windowName, screenWithOverlay);
	}
	
	Vec4f test(0.5f, 0.5f, 0.5f, 1.0f);
	
	return 0;
}

