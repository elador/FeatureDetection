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

#include "imageio/LandmarkCollection.hpp"
#include "imageio/ModelLandmark.hpp"
#include "imageio/DidLandmarkFormatParser.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"

#include <iostream>
#include <fstream>
#include <random>
#include <functional>

namespace po = boost::program_options;
using namespace std;
using namespace cv;
using namespace imageio;
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

	//render::Mesh morphableModel = render::utils::MeshUtils::readFromHdf5("D:\\model2012_l6_rms.h5");
	render::MorphableModel morphableModel = render::utils::MeshUtils::readFromScm("C:\\Users\\Patrik\\Cloud\\PhD\\MorphModel\\ShpVtxModelBin.scm");
	render::Mesh cube = render::utils::MeshUtils::createCube();
	render::Mesh pyramid = render::utils::MeshUtils::createPyramid();
	render::Mesh plane = render::utils::MeshUtils::createPlane();

	int screenWidth = 640;
	int screenHeight = 480;
	const float aspect = (float)screenWidth/(float)screenHeight;

	//Camera camera(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));
	Camera camera(Vec3f(0.0f, 0.0f, 0.0f), horizontalAngle*(CV_PI/180.0f), verticalAngle*(CV_PI/180.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));

	RenderDevice r(screenWidth, screenHeight, camera);

	namedWindow(windowName, WINDOW_AUTOSIZE);
	setMouseCallback(windowName, winOnMouse);

	vector<int> vertexIds;
	vertexIds.push_back(177); // left-eye-left - right.eye.corner_outer
	vertexIds.push_back(181); // left-eye-right - right.eye.corner_inner
	vertexIds.push_back(614); // right-eye-left - left.eye.corner_inner
	vertexIds.push_back(610); // right-eye-right - left.eye.corner_outer
	vertexIds.push_back(398); // mouth-left - right.lips.corner
	vertexIds.push_back(812); // mouth-right - left.lips.corner
	vertexIds.push_back(11140); // bridge of the nose - 
	vertexIds.push_back(114); // nose-tip - center.nose.tip
	vertexIds.push_back(270); // nasal septum - 
	vertexIds.push_back(3284); // left-alare - right nose ...
	vertexIds.push_back(572); // right-alare - left nose ...

	ofstream outputFile;
	outputFile.open("C:/Users/Patrik/Cloud/data_3dmm_landmarks_norm02_batch3_y.txt");
	outputFile << "pose_rndvtx_x " << "pose_rndvtx_y " << "frontal_rndvtx_x " << "frontal_rndvtx_y " << "pose_lel_x " << "pose_lel_y " << "pose_ler_x " << "pose_ler_y " << "pose_rel_x " << "pose_rel_y " << "pose_rer_x " << "pose_rer_y " << "pose_ml_x " << "pose_ml_y " << "pose_mr_x " << "pose_mr_y " << "pose_bn_x " << "pose_bn_y " << "pose_nt_x " << "pose_nt_y " << "pose_ns_x " << "pose_ns_y " << "pose_la_x " << "pose_la_y " << "pose_ra_x " << "pose_ra_y " << "yaw " << "pitch " << "roll" << std::endl;
	
	
	std::uniform_int_distribution<int> distribution(0, morphableModel.mesh.vertex.size()-1);
	std::mt19937 engine; // Mersenne twister MT19937
	//std::random_device rd;
	//engine.seed(rd());
	engine.seed();
	auto randInt = std::bind(distribution, engine);
	
	float xAngle = 0.0f;
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
			xAngle -= 1.0f;
		}
		if (key == 'm') {
			xAngle += 1.0f;
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

		r.setWorldTransform(render::utils::MatrixUtils::createScalingMatrix(1.0f, 1.0f, 1.0f));
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

		for (int angle = -45; angle <= 45; ++angle) {
			for (int sampleNumPerDegree = 0; sampleNumPerDegree < 200; ++sampleNumPerDegree) {
				r.resetBuffers();

				morphableModel.drawNewVertexPositions();

				int randomVertex = randInt();
				vector<shared_ptr<Landmark>> pointsToWrite;
				
				// 1) Render the randomVertex frontal
				Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f);
				Mat rot = Mat::eye(4, 4, CV_32FC1);
				Mat modelMatrix = rot * modelScaling;
				r.setWorldTransform(modelMatrix);
				Vec2f res = r.projectVertex(morphableModel.mesh.vertex[randomVertex].position);
				string name = "randomVertexFrontal";
				pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));
				
				// 2) Render the randomVertex in pose angle
				rot = render::utils::MatrixUtils::createRotationMatrixY(angle * (CV_PI/180.0f));
				modelMatrix = rot * modelScaling;
				r.setWorldTransform(modelMatrix);
				res = r.projectVertex(morphableModel.mesh.vertex[randomVertex].position);
				name = "randomVertexPose";
				pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));

				// 3) Render all LMs in pose angle
				for (const auto& vid : vertexIds) {
					rot = render::utils::MatrixUtils::createRotationMatrixY(angle * (CV_PI/180.0f));
					modelMatrix = rot * modelScaling;
					r.setWorldTransform(modelMatrix);
					res = r.projectVertex(morphableModel.mesh.vertex[vid].position);
					//r.renderLM(morphableModel.mesh.vertex[vid].position, Scalar(255.0f, 0.0f, 0.0f));
					name = DidLandmarkFormatParser::didToTlmsName(vid);
					pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));
				}

				Mat screen = r.getImage();
				// 4) Write one row to the file
				for (const auto& lm : pointsToWrite) {
					lm->draw(screen);
					outputFile << lm->getX() << " " << lm->getY() << " ";
				}
				int yaw = angle;
				int pitch = 0;
				int roll = 0;
				outputFile << yaw << " " << pitch << " " << roll << std::endl;
				

			}
		}

		outputFile.close();

		Mat screen = r.getImage();
		/*
		for (const auto& lm : lms) {
			lm->draw(screen);
		}*/

		putText(screen, "(" + lexical_cast<string>(lastX) + ", " + lexical_cast<string>(lastY) + ")", Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "horA: " + lexical_cast<string>(horizontalAngle) + ", verA: " + lexical_cast<string>(verticalAngle), Point(10, 38), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "moving: " + lexical_cast<string>(moving), Point(10, 56), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "zNear: " + lexical_cast<string>(r.camera.frustum.n) + ", zFar: " + lexical_cast<string>(r.camera.frustum.f) + ", p: " + lexical_cast<string>(perspective), Point(10, 74), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		putText(screen, "eye: " + lexical_cast<string>(r.camera.eye[0]) + ", " + lexical_cast<string>(r.camera.eye[1]) + ", " + lexical_cast<string>(r.camera.eye[2]), Point(10, 92), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 0, 255));
		imshow(windowName, screen);

	}

	/*
	for (int angle = -45; angle <= 45; ++angle) {
		vector<shared_ptr<Landmark>> lms;

		for (const auto& vid : vertexIds) {
			r.renderDevice->setWorldTransform(render::utils::MatrixUtils::createRotationMatrixX(angle * (CV_PI/180.0f)));
			Vec2f res = r.renderDevice->renderVertex(morphableModel.mesh.vertex[vid].position);
			string name = DidLandmarkFormatParser::didToTlmsName(vid);
			shared_ptr<Landmark> lm = make_shared<ModelLandmark>(name, res);
			lms.push_back(lm);
		}

		Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
		for (const auto& lm : lms) {
			lm->draw(img);
		}
	}*/



	return 0;
}

// TODO: All those vec3 * mat4 cases... Do I have to add the homogeneous coordinate and make it vec4 * mat4, instead of how I'm doing it now? Difference?