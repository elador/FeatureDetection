// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#ifdef WIN32
	#include <crtdbg.h>
#endif

/*#ifdef _DEBUG
   #ifndef DBG_NEW
	  #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
	  #define new DBG_NEW
   #endif
#endif  // _DEBUG
   */
#include "render/MeshUtils.hpp"
#include "render/MatrixUtils.hpp"
#include "render/SoftwareRenderer.hpp"
#include "render/Camera.hpp"

#include "morphablemodel/MorphableModel.hpp"

#include "logging/LoggerFactory.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"

#include <iostream>
#include <fstream>

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using namespace std;
using namespace cv;
using namespace render;
using boost::lexical_cast;
using boost::property_tree::ptree;
using boost::filesystem::path;

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
	return os;
}

static string controlWindowName = "Controls";
static std::array<float, 55> pcVals;
static std::array<int, 55> pcValsInt;

static morphablemodel::MorphableModel mm;
static shared_ptr<Mesh> meshToDraw;

static string windowName = "RenderOutput";
static float horizontalAngle = 0.0f;
static float verticalAngle = 0.0f;
static int lastX = 0;
static int lastY = 0;
static float movingFactor = 0.5f;
static bool moving = false;

static float zNear = -0.1f;
static float zFar = -100.0f;
static bool perspective = false;
static bool freeCamera = true;

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

static void controlWinOnMouse(int test, void* userdata)
{
	// TODO: This breaks when the has less than 55 PCA comps. Make more dynamic...
	for (int i = 0; i < pcValsInt.size(); ++i) {
		pcVals[i] = 0.1f * static_cast<float>(pcValsInt[i]) - 5.0f;
	}
	Mat coefficients = Mat::zeros(mm.getShapeModel().getNumberOfPrincipalComponents(), 1, CV_32FC1);
	for (int row=0; row < pcVals.size(); ++row) {
		coefficients.at<float>(row, 0) = pcVals[row];
	}

	meshToDraw.reset();
	meshToDraw = make_shared<Mesh>(mm.drawSample(coefficients, vector<float>()));
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF); // dump leaks at return
	//_CrtSetBreakAlloc(3759128);
	#endif

	path configFilename;
	
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("config,c", po::value<path>(&configFilename)->required(),
			"path to a config (.cfg) file")
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

	}
	catch (std::exception& e) {
		cout << e.what() << endl;
		return 1;
	}
	
	Loggers->getLogger("morphablemodel").addAppender(std::make_shared<logging::ConsoleAppender>(LogLevel::Trace));
	Logger appLogger = Loggers->getLogger("3dmmRendererGUI");
	appLogger.addAppender(std::make_shared<logging::ConsoleAppender>(LogLevel::Trace));

	render::Mesh cube = render::utils::MeshUtils::createCube();
	render::Mesh pyramid = render::utils::MeshUtils::createPyramid();
	render::Mesh plane = render::utils::MeshUtils::createPlane();
	shared_ptr<render::Mesh> tri = render::utils::MeshUtils::createTriangle();
	
	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(configFilename.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	morphablemodel::MorphableModel mm;
	try {
		mm = morphablemodel::MorphableModel::load(pt.get_child("morphableModel"));
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	catch (const std::runtime_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	meshToDraw = std::make_shared<Mesh>(mm.getMean());

	int screenWidth = 640;
	int screenHeight = 480;
	const float aspect = (float)screenWidth/(float)screenHeight;

	//Camera camera(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));
	Camera camera(Vec3f(0.0f, 0.0f, 0.0f), horizontalAngle*(CV_PI/180.0f), verticalAngle*(CV_PI/180.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));

	SoftwareRenderer r(screenWidth, screenHeight, camera);

	namedWindow(windowName, WINDOW_AUTOSIZE);
	setMouseCallback(windowName, winOnMouse);

	namedWindow(controlWindowName, WINDOW_NORMAL);
	for (auto& val : pcValsInt)	{
		val = 50;
	}
	for (unsigned int i = 0; i < pcVals.size(); ++i) {
		createTrackbar("PC" + lexical_cast<string>(i) + ": " + lexical_cast<string>(pcVals[i]), controlWindowName, &pcValsInt[i], 100, controlWinOnMouse);
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
		if (key == 'w') {
			r.camera.eye += 0.05f * -camera.getForwardVector();
		}
		if (key == 'a') {
			r.camera.eye -= 0.05f * camera.getRightVector();
		}
		if (key == 's') {
			r.camera.eye -= 0.05f * -camera.getForwardVector();
		}
		if (key == 'd') {
			r.camera.eye += 0.05f * camera.getRightVector();
		}
		if (key == 'r') {
			r.camera.eye += 0.05f * camera.getUpVector();
		}
		if (key == 'f') {
			r.camera.eye -= 0.05f * camera.getUpVector();
		}

		if (key == 'z') {
			r.camera.gaze += 0.05f * camera.getUpVector();
		}
		if (key == 'h') {
			r.camera.gaze -= 0.05f * camera.getUpVector();
		}
		if (key == 'g') {
			r.camera.gaze -= 0.05f * camera.getRightVector();
		}
		if (key == 'j') {
			r.camera.gaze += 0.05f * camera.getRightVector();
		}

		if (key == 'i') {
			r.camera.frustum.n += 0.1f;
		}
		if (key == 'k') {
			r.camera.frustum.n -= 0.1f;
		}
		if (key == 'o') {
			r.camera.frustum.f += 1.0f;
		}
		if (key == 'l') {
			r.camera.frustum.f -= 1.0f;
		}
		if (key == 'p') {
			perspective = !perspective;
		}
		if (key == 'c') {
			freeCamera = !freeCamera;
		}
		if (key == 'n') {
			meshToDraw.reset();
			meshToDraw = make_shared<Mesh>(mm.drawSample(0.7f));
		}
		
		r.resetBuffers();

		r.camera.horizontalAngle = horizontalAngle*(CV_PI/180.0f);
		r.camera.verticalAngle = verticalAngle*(CV_PI/180.0f);
		if(freeCamera) {
			r.camera.updateFree(r.camera.eye);
		} else {
			r.camera.updateFixed(r.camera.eye, r.camera.gaze);
		}

		r.updateViewTransform();
		r.updateProjectionTransform(perspective);

		Vec4f origin(0.0f, 0.0f, 0.0f, 1.0f);
		Vec4f xAxis(1.0f, 0.0f, 0.0f, 1.0f);
		Vec4f yAxis(0.0f, 1.0f, 0.0f, 1.0f);
		Vec4f zAxis(0.0f, 0.0f, 1.0f, 1.0f);
		Vec4f mzAxis(0.0f, 0.0f, -1.0f, 1.0f);
		r.renderLine(origin, xAxis, Scalar(0.0f, 0.0f, 255.0f));
		r.renderLine(origin, yAxis, Scalar(0.0f, 255.0f, 0.0f));
		r.renderLine(origin, zAxis, Scalar(255.0f, 0.0f, 0.0f));
		r.renderLine(origin, mzAxis, Scalar(0.0f, 255.0f, 255.0f));
		/*
		for (const auto& tIdx : cube.tvi) {
			Vertex v0 = cube.vertex[tIdx[0]];
			Vertex v1 = cube.vertex[tIdx[1]];
			Vertex v2 = cube.vertex[tIdx[2]];
			r.renderLine(v0.position, v1.position, Scalar(0.0f, 0.0f, 255.0f));
			r.renderLine(v1.position, v2.position, Scalar(0.0f, 0.0f, 255.0f));
			r.renderLine(v2.position, v0.position, Scalar(0.0f, 0.0f, 255.0f));
		}

		// Test-vertex for rasterizing:
		Vertex v0 = cube.vertex[0];
		Vertex v1 = cube.vertex[1];
		Vertex v2 = cube.vertex[2];
		r.renderLine(v0.position, v1.position, Scalar(255.0f, 0.0f, 0.0f));
		r.renderLine(v1.position, v2.position, Scalar(255.0f, 0.0f, 0.0f));
		r.renderLine(v2.position, v0.position, Scalar(255.0f, 0.0f, 0.0f));
		*/
		//r.draw(tri, tri->texture);
		
		Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f);
		//Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(7.0f, 7.0f, 7.0f);
		Mat modelTrans = render::utils::MatrixUtils::createTranslationMatrix(0.0f, 0.0f, -1.5f);
		Mat rot = Mat::eye(4, 4, CV_32FC1);
		Mat modelMatrix = modelTrans * rot * modelScaling;
		r.setModelTransform(modelMatrix);
		r.draw(meshToDraw, nullptr);
		
		r.setModelTransform(Mat::eye(4, 4, CV_32FC1));
		// End test
		
		//r.renderLine(Vec4f(1.5f, 0.0f, 0.5f, 1.0f), Vec4f(-1.5f, 0.0f, 0.5f, 1.0f), Scalar(0.0f, 0.0f, 255.0f));
		Mat zBuffer = r.getDepthBuffer();
		Mat screen = r.getImage();
		render::Mesh::writeObj(*meshToDraw.get(), "C:\\Users\\Patrik\\Documents\\GitHub\\test_mean.obj");
		putText(screen, "(" + lexical_cast<string>(lastX) + ", " + lexical_cast<string>(lastY) + ")", Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "horA: " + lexical_cast<string>(horizontalAngle) + ", verA: " + lexical_cast<string>(verticalAngle), Point(10, 38), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "moving: " + lexical_cast<string>(moving), Point(10, 56), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "zNear: " + lexical_cast<string>(r.camera.frustum.n) + ", zFar: " + lexical_cast<string>(r.camera.frustum.f) + ", p: " + lexical_cast<string>(perspective), Point(10, 74), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "eye: " + lexical_cast<string>(r.camera.eye[0]) + ", " + lexical_cast<string>(r.camera.eye[1]) + ", " + lexical_cast<string>(r.camera.eye[2]), Point(10, 92), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		imshow(windowName, screen);

	}
	
	return 0;
}

