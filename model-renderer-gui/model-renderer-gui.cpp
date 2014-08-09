// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>

#ifdef WIN32
	//#include <crtdbg.h>
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
#include "render/utils.hpp"
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

#include "QMatrix4x4"
#include "QDebug"
#include "QtMath"

#include <iostream>
#include <fstream>

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using namespace render;
using cv::Mat;
using cv::Scalar;
using cv::Point;
using cv::Vec4f;
using boost::lexical_cast;
using boost::property_tree::ptree;
using boost::filesystem::path;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::make_shared;

float radiansToDegrees(float radians) // change to constexpr in VS2014
{
	return radians * static_cast<float>(180 / CV_PI);
}

float degreesToRadians(float degrees) // change to constexpr in VS2014
{
	return degrees * static_cast<float>(CV_PI / 180);
}

template<class T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v)
{
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
	return os;
}

static string controlWindowName = "Controls";
static std::array<float, 55> pcVals;
static std::array<int, 55> pcValsInt;

//static morphablemodel::MorphableModel mm;
//static shared_ptr<Mesh> meshToDraw;
Mesh meshToDraw;

static string windowName = "RenderOutput";
static float horizontalAngle = 0.0f;
static float verticalAngle = 0.0f;
static int lastX = 0;
static int lastY = 0;
static float movingFactor = 0.5f;
static bool moving = false;

static float zNear = 0.1f;
static float zFar = 100.0f;
static bool perspective = false;
static bool freeCamera = true;

static void winOnMouse(int event, int x, int y, int, void* userdata)
{
	if (event == cv::EVENT_MOUSEMOVE && moving == false) {
		lastX = x;
		lastY = y;
	}
	if (event == cv::EVENT_MOUSEMOVE && moving == true) {
		if (x != lastX || y != lastY) {
			horizontalAngle += (x-lastX)*movingFactor;
			verticalAngle += (y-lastY)*movingFactor;
			lastX = x;
			lastY = y;
		}
	}
	if (event == cv::EVENT_LBUTTONDOWN) {
		moving = true;
		lastX = x;
		lastY = y;
	}
	if (event == cv::EVENT_LBUTTONUP) {
		moving = false;
	}

}

static void controlWinOnMouse(int test, void* userdata)
{
	// TODO: This breaks when the has less than 55 PCA comps. Make more dynamic...
	for (int i = 0; i < pcValsInt.size(); ++i) {
		pcVals[i] = 0.1f * static_cast<float>(pcValsInt[i]) - 5.0f;
	}
	/*
	Mat coefficients = Mat::zeros(mm.getShapeModel().getNumberOfPrincipalComponents(), 1, CV_32FC1);
	for (int row=0; row < pcVals.size(); ++row) {
		coefficients.at<float>(row, 0) = pcVals[row];
	}
	*/
	//meshToDraw.reset();
	//meshToDraw = make_shared<Mesh>(mm.drawSample(coefficients, vector<float>()));
	//meshToDraw = mm.drawSample(coefficients, vector<float>());
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF); // dump leaks at return
	//_CrtSetBreakAlloc(3759128);
	#endif

	string verboseLevelConsole;
	path morphableModelConfigFilename;
	
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO", "show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("model,m", po::value<path>(&morphableModelConfigFilename)->required(),
				"path to a config file that specifies which Morphable Model to load")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: model-renderer-gui [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);

	}
	catch (po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	LogLevel logLevel;
	if (boost::iequals(verboseLevelConsole, "PANIC")) logLevel = LogLevel::Panic;
	else if (boost::iequals(verboseLevelConsole, "ERROR")) logLevel = LogLevel::Error;
	else if (boost::iequals(verboseLevelConsole, "WARN")) logLevel = LogLevel::Warn;
	else if (boost::iequals(verboseLevelConsole, "INFO")) logLevel = LogLevel::Info;
	else if (boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = LogLevel::Debug;
	else if (boost::iequals(verboseLevelConsole, "TRACE")) logLevel = LogLevel::Trace;
	else {
		cout << "Error: Invalid log level." << endl;
		return EXIT_SUCCESS;
	}
	
	Loggers->getLogger("morphablemodel").addAppender(std::make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("model-renderer-gui").addAppender(std::make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("model-renderer-gui");
	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	render::Mesh mesh = render::utils::MeshUtils::createCube();
	//render::Mesh pyramid = render::utils::MeshUtils::createPyramid();
	//render::Mesh plane = render::utils::MeshUtils::createPlane();
	//shared_ptr<render::Mesh> tri = render::utils::MeshUtils::createTriangle();
	/*
	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(morphableModelConfigFilename.string(), pt);
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
	*/
	//meshToDraw = mm.getMean();
	meshToDraw = mesh;

	int screenWidth = 640;
	int screenHeight = 480;
	const float aspect = static_cast<float>(screenWidth) / static_cast<float>(screenHeight);

	Mat moveCameraBack = render::utils::MatrixUtils::createTranslationMatrix(0.0f, 0.0f, -3.0f);
	Mat projection = render::utils::MatrixUtils::createOrthogonalProjectionMatrix(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar);
	//Mat projection = render::utils::MatrixUtils::createPerspectiveProjectionMatrix(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar);

	//Camera camera(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));
	Camera camera(Vec3f(0.0f, 0.0f, 0.0f), degreesToRadians(horizontalAngle), degreesToRadians(verticalAngle), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));

	SoftwareRenderer r(screenWidth, screenHeight);

	using namespace render::utils;
	//Mat cm = MatrixUtils::createOrthogonalProjectionMatrix(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, -0.1f, -100.0f);
	
	namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback(windowName, winOnMouse);

	namedWindow(controlWindowName, cv::WINDOW_NORMAL);
	for (auto& val : pcValsInt)	{
		val = 50;
	}
	for (unsigned int i = 0; i < pcVals.size(); ++i) {
		cv::createTrackbar("PC" + lexical_cast<string>(i)+": " + lexical_cast<string>(pcVals[i]), controlWindowName, &pcValsInt[i], 100, controlWinOnMouse);
	}
	

	bool running = true;
	while (running) {
		int key = cv::waitKey(30);
		if (key==-1) {
			// no key pressed
		}
		if (key == 'q') {
			running = false;
		}
		if (key == 'w') {
			camera.moveForward(0.05f);
		}
		if (key == 'a') {
			camera.moveRight(-0.05f);
		}
		if (key == 's') {
			camera.moveForward(-0.05f);
		}
		if (key == 'd') {
			camera.moveRight(0.05f);
		}
		if (key == 'r') {
			camera.moveUp(0.05f);
		}
		if (key == 'f') {
			camera.moveUp(-0.05f);
		}

		if (key == 'z') {
			camera.rotateUp(0.05f);
		}
		if (key == 'h') {
			camera.rotateUp(-0.05f);
		}
		if (key == 'g') {
			camera.rotateRight(-0.05f);
		}
		if (key == 'j') {
			camera.rotateRight(0.05f);
		}

		if (key == 'i') {
			camera.getFrustum().n += 0.1f;
		}
		if (key == 'k') {
			camera.getFrustum().n -= 0.1f;
		}
		if (key == 'o') {
			camera.getFrustum().f += 1.0f;
		}
		if (key == 'l') {
			camera.getFrustum().f -= 1.0f;
		}
		if (key == 'p') {
			perspective = !perspective;
		}
		if (key == 'c') {
			freeCamera = !freeCamera;
		}
		if (key == 'n') {
			//meshToDraw.reset();
			//meshToDraw = make_shared<Mesh>(mm.drawSample(0.7f));
			//meshToDraw = mm.drawSample(0.7f);
		}
		
		//r.resetBuffers();

		if(freeCamera) {
			// remove eye as arg, rename to ... setGazeDirection()? and add a setPosition()?
			camera.updateFree(camera.eye, degreesToRadians(horizontalAngle), degreesToRadians(verticalAngle));
		} else {
			camera.updateFixed(camera.eye, camera.gaze);
		}

		//r.updateViewTransform();
		//r.updateProjectionTransform(perspective);
		// => moved to render()
		
		if (perspective) {
			camera.projectionType = Camera::ProjectionType::Perspective;
		}
		else {
			camera.projectionType = Camera::ProjectionType::Orthogonal;
		}
		//projection = render::utils::MatrixUtils::createOrthogonalProjectionMatrix(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, camera.frustum.n, camera.frustum.f);
		projection = camera.getProjectionMatrix();


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
		
		//Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f);
		Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(1.0f, 1.0f, 1.0f);
		Mat modelTrans = render::utils::MatrixUtils::createTranslationMatrix(0.0f, 0.0f, -3.0f);
		//Mat rot = Mat::eye(4, 4, CV_32FC1);
		Mat rot = render::utils::MatrixUtils::createRotationMatrixX(degreesToRadians(20.0f)) * render::utils::MatrixUtils::createRotationMatrixY(degreesToRadians(20.0f));
		Mat modelMatrix = modelTrans * rot * modelScaling;
		//r.setModelTransform(modelMatrix);
		//r.draw(meshToDraw, nullptr);
		
		//r.setModelTransform(Mat::eye(4, 4, CV_32FC1));
		
		// End test
		
		//r.renderLine(Vec4f(1.5f, 0.0f, 0.5f, 1.0f), Vec4f(-1.5f, 0.0f, 0.5f, 1.0f), Scalar(0.0f, 0.0f, 255.0f));
		//Mat zBuffer = r.getDepthBuffer();
		//Mat screen = r.getImage();

		Mat cameraTransform = camera.getViewMatrix();

		Mat zBuffer, screen;
		r.clearBuffers();
		std::tie(screen, zBuffer) = r.render(meshToDraw, projection * cameraTransform * modelMatrix);

		// Draw the 3 axes over the screen
		// Note: We should use render(), so that the depth buffer gets used?
		Vec4f origin(0.0f, 0.0f, 0.0f, 1.0f);
		Vec4f xAxis(1.0f, 0.0f, 0.0f, 1.0f);
		Vec4f yAxis(0.0f, 1.0f, 0.0f, 1.0f);
		Vec4f zAxis(0.0f, 0.0f, 1.0f, 1.0f);
		Vec4f mzAxis(0.0f, 0.0f, -1.0f, 1.0f);
		Vec3f originScreen = render::utils::projectVertex(origin, projection * modelMatrix, screenWidth, screenHeight);
		Vec3f xAxisScreen = render::utils::projectVertex(xAxis, projection * modelMatrix, screenWidth, screenHeight);
		Vec3f yAxisScreen = render::utils::projectVertex(yAxis, projection * modelMatrix, screenWidth, screenHeight);
		Vec3f zAxisScreen = render::utils::projectVertex(zAxis, projection * modelMatrix, screenWidth, screenHeight);
		Vec3f mzAxisScreen = render::utils::projectVertex(mzAxis, projection * modelMatrix, screenWidth, screenHeight);
		cv::line(screen, cv::Point(originScreen[0], originScreen[1]), cv::Point(xAxisScreen[0], xAxisScreen[1]), Scalar(0.0f, 0.0f, 255.0f, 255.0f));
		cv::line(screen, cv::Point(originScreen[0], originScreen[1]), cv::Point(yAxisScreen[0], yAxisScreen[1]), Scalar(0.0f, 255.0f, 0.0f, 255.0f));
		cv::line(screen, cv::Point(originScreen[0], originScreen[1]), cv::Point(zAxisScreen[0], zAxisScreen[1]), Scalar(255.0f, 0.0f, 0.0f, 255.0f));
		cv::line(screen, cv::Point(originScreen[0], originScreen[1]), cv::Point(mzAxisScreen[0], mzAxisScreen[1]), Scalar(0.0f, 255.0f, 255.0f, 255.0f));

		render::Mesh::writeObj(meshToDraw, "C:\\Users\\Patrik\\Documents\\GitHub\\test_asdf.obj");
		putText(screen, "(" + lexical_cast<string>(lastX) + ", " + lexical_cast<string>(lastY) + ")", Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "horA: " + lexical_cast<string>(horizontalAngle) + ", verA: " + lexical_cast<string>(verticalAngle), Point(10, 38), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "moving: " + lexical_cast<string>(moving), Point(10, 56), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "zNear: " + lexical_cast<string>(camera.getFrustum().n) + ", zFar: " + lexical_cast<string>(camera.getFrustum().f) + ", p: " + lexical_cast<string>(perspective), Point(10, 74), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "eye: " + lexical_cast<string>(camera.eye[0]) + ", " + lexical_cast<string>(camera.eye[1]) + ", " + lexical_cast<string>(camera.eye[2]), Point(10, 92), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		imshow(windowName, screen);

	}
	
	return 0;
}

