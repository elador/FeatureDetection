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

#include "render/MeshUtils.hpp"
#include "render/MatrixUtils.hpp"
#include "render/SoftwareRenderer.hpp"
#include "render/Camera.hpp"

#include "shapemodels/MorphableModel.hpp"

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

static float zNear = -0.01f;
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

	//mm = shapemodels::MorphableModel::loadScmModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\SurreyLowResGuosheng\\NON3448\\ShpVtxModelBin_NON3448.scm", "C:\\Users\\Patrik\\Documents\\GitHub\\featurePoints_SurreyScm.txt");
	//mm = shapemodels::MorphableModel::loadOldBaselH5Model("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\model2012p.h5", "featurePoints_head_newfmt.txt");
	//mm = shapemodels::MorphableModel::loadStatismoModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\2012.2\\head\\model2012_l7_head.h5");
	shapemodels::MorphableModel morphableModel = shapemodels::MorphableModel::loadScmModel("C:\\Users\\Patrik\\Cloud\\PhD\\MorphModel\\ShpVtxModelBin.scm", "C:\\Users\\Patrik\\Documents\\GitHub\\featurePoints_SurreyScm.txt");
	//shapemodels::MorphableModel morphableModel = shapemodels::MorphableModel::loadScmModel(".\\ShpVtxModelBin.scm", ".\\featurePoints_SurreyScm.txt");
	//render::Mesh cube = render::utils::MeshUtils::createCube();
	//render::Mesh pyramid = render::utils::MeshUtils::createPyramid();
	//render::Mesh plane = render::utils::MeshUtils::createPlane();

	int screenWidth = 640;
	int screenHeight = 480;
	const float aspect = (float)screenWidth/(float)screenHeight;

	//Camera camera(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));
	Camera camera(Vec3f(0.0f, 0.0f, 3.0f), horizontalAngle*(CV_PI/180.0f), verticalAngle*(CV_PI/180.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));

	SoftwareRenderer r(screenWidth, screenHeight, camera);

	//namedWindow(windowName, WINDOW_AUTOSIZE);
	//setMouseCallback(windowName, winOnMouse);

	vector<int> vertexIds; // Todo: Work with the tlms names everywhere instead of the vertex indices
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
	//outputFile.open("C:/Users/Patrik/Documents/Github/syndata/data_3dmm_landmarks_random_sd0.5_batch4_600000k.txt");
	outputFile.open("./data_3dmm_landmarks_random_sd0.5_batch5_600000k.txt");
	outputFile << "frontal_rndvtx_x " << "frontal_rndvtx_y " << "frontal_lel_x " << "frontal_lel_y " << "frontal_ler_x " << "frontal_ler_y " << "frontal_rel_x " << "frontal_rel_y " << "frontal_rer_x " << "frontal_rer_y " << "frontal_ml_x " << "frontal_ml_y " << "frontal_mr_x " << "frontal_mr_y " << "frontal_bn_x " << "frontal_bn_y " << "frontal_nt_x " << "frontal_nt_y " << "frontal_ns_x " << "frontal_ns_y " << "frontal_la_x " << "frontal_la_y " << "frontal_ra_x " << "frontal_ra_y " << "pose_rndvtx_x " << "pose_rndvtx_y " << "pose_lel_x " << "pose_lel_y " << "pose_ler_x " << "pose_ler_y " << "pose_rel_x " << "pose_rel_y " << "pose_rer_x " << "pose_rer_y " << "pose_ml_x " << "pose_ml_y " << "pose_mr_x " << "pose_mr_y " << "pose_bn_x " << "pose_bn_y " << "pose_nt_x " << "pose_nt_y " << "pose_ns_x " << "pose_ns_y " << "pose_la_x " << "pose_la_y " << "pose_ra_x " << "pose_ra_y " << "yaw " << "pitch " << "roll " << "rndvtx_id" << std::endl;
	
	int numVertices = morphableModel.getShapeModel().getDataDimension() / 3;
	std::uniform_int_distribution<int> distribution(0, numVertices-1);
	std::mt19937 engine; // Mersenne twister MT19937
	//std::random_device rd;
	//engine.seed(rd());
	engine.seed();
	auto randIntVtx = std::bind(distribution, engine);

	std::uniform_int_distribution<int> distrRandYaw(-45, 45);
	auto randIntYaw = std::bind(distrRandYaw, engine);
	std::uniform_int_distribution<int> distrRandPitch(-30, 30);
	auto randIntPitch = std::bind(distrRandPitch, engine);
	//std::uniform_int_distribution<int> distrRandRoll(-20, 20);
	std::uniform_int_distribution<int> distrRandRoll(0, 0);
	auto randIntRoll = std::bind(distrRandRoll, engine);
	
	r.resetBuffers();

	r.camera.horizontalAngle = horizontalAngle*(CV_PI/180.0f);
	r.camera.verticalAngle = verticalAngle*(CV_PI/180.0f);
	r.camera.updateFixed(r.camera.eye, r.camera.gaze);

	r.updateViewTransform();
	r.updateProjectionTransform(perspective);

	int generatedSamples = 0;
	int samplesToGenerate = 600000; //600000;
	while (generatedSamples < samplesToGenerate) {
		r.resetBuffers();
		std::cout << generatedSamples << std::endl;

		render::Mesh newSampleMesh = morphableModel.drawSample(0.5f); // Note: it would suffice to only draw a shape model, but then we can't render it

		int randomVertex = randIntVtx();
		int yaw = randIntYaw();
		int pitch = randIntPitch();
		int roll = randIntRoll();

		vector<shared_ptr<Landmark>> pointsToWrite;
				
		// 1) Render the randomVertex frontal
		Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f);
		Mat rotPitchX = Mat::eye(4, 4, CV_32FC1);
		Mat rotYawY = Mat::eye(4, 4, CV_32FC1);
		Mat rotRollZ = Mat::eye(4, 4, CV_32FC1);
		Mat modelMatrix = rotYawY * rotPitchX * rotRollZ * modelScaling;
		r.setModelTransform(modelMatrix);
		Vec2f res = r.projectVertex(newSampleMesh.vertex[randomVertex].position);
		string name = "randomVertexFrontal";
		pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));

		// 2) Render all LMs in frontal pose
		for (const auto& vid : vertexIds) {
			modelMatrix = rotYawY * rotPitchX * rotRollZ * modelScaling; // same as before in 1)
			r.setModelTransform(modelMatrix);
			res = r.projectVertex(newSampleMesh.vertex[vid].position);
			//r.renderLM(newSampleMesh.vertex[vid].position, Scalar(255.0f, 0.0f, 0.0f));
			name = DidLandmarkFormatParser::didToTlmsName(vid);
			pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));
		}

		// 3) Render the randomVertex in pose angle
		rotPitchX = render::utils::MatrixUtils::createRotationMatrixX(pitch * (CV_PI/180.0f));
		rotYawY = render::utils::MatrixUtils::createRotationMatrixY(yaw * (CV_PI/180.0f));
		rotRollZ = render::utils::MatrixUtils::createRotationMatrixZ(roll * (CV_PI/180.0f));
		modelMatrix = rotYawY * rotPitchX * rotRollZ * modelScaling;
		r.setModelTransform(modelMatrix);
		// tmp:
		shared_ptr<Mesh> mmMesh = std::make_shared<Mesh>(newSampleMesh);
		r.draw(mmMesh, nullptr);
		Mat screen = r.getImage();
		Mat depth = r.getDepthBuffer();

		pair<Vec2f, bool> vertexVisPair;
		//res = r.projectVertex(newSampleMesh.vertex[randomVertex].position);
		vertexVisPair = r.projectVertexVis(newSampleMesh.vertex[randomVertex].position);
		cv::circle(screen, cv::Point(vertexVisPair.first[0], vertexVisPair.first[1]), 3, cv::Scalar(255, 0, 128));
		if (vertexVisPair.second != true) {
			continue;
		}
		res = vertexVisPair.first;
		name = "randomVertexPose";
		pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));

		// 4) Render all LMs in pose angle
		for (const auto& vid : vertexIds) {
			modelMatrix = rotYawY * rotPitchX * rotRollZ * modelScaling; // same as before in 3)
			r.setModelTransform(modelMatrix);
			res = r.projectVertex(newSampleMesh.vertex[vid].position);
			//r.renderLM(morphableModel.mesh.vertex[vid].position, Scalar(255.0f, 0.0f, 0.0f));
			name = DidLandmarkFormatParser::didToTlmsName(vid);
			pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));
		}
					
		// 4) Write one row to the file
		for (const auto& lm : pointsToWrite) {
			//lm->draw(screen);
			outputFile << lm->getX() << " " << lm->getY() << " ";
		}

		outputFile << yaw << " " << pitch << " " << roll << " " << randomVertex << std::endl;
		++generatedSamples;
	}

	outputFile.close();

	return 0;
}

// TODO: All those vec3 * mat4 cases... Do I have to add the homogeneous coordinate and make it vec4 * mat4, instead of how I'm doing it now? Difference?