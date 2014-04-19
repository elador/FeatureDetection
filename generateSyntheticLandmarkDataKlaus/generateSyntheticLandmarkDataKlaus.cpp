// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#ifdef WIN32
	#include <crtdbg.h>
#endif

/* // doesn't work with boost 1.55.0 on debug compile
// try: http://msdn.microsoft.com/en-us/library/tz7sxz99.aspx ?
#ifdef _DEBUG
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

#include "imageio/LandmarkCollection.hpp"
#include "imageio/ModelLandmark.hpp"
#include "imageio/DidLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"

#include <iostream>
#include <fstream>
#include <random>
#include <functional>

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using namespace std;
using namespace cv;
using namespace imageio;
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

static string windowName = "test";
static float horizontalAngle = 0.0f;
static float verticalAngle = 0.0f;
static int lastX = 0;
static int lastY = 0;
static float movingFactor = 0.5f;
static bool moving = false;

static float zNear = 0.01f;
static float zFar = 100.0f;
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

	//std::string filename; // Create vector to hold the filenames
	path configFilename;
	
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("config,c", po::value<path>(&configFilename)->required(),
			"Path to a config file that specifies which Morphable Model to load.")
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
	catch(std::exception& e) {
		cout << e.what() << "\n";
		return 1;
	}

	Logger appLogger = Loggers->getLogger("morphablemodel");
	appLogger.addAppender(std::make_shared<logging::ConsoleAppender>(LogLevel::Trace));

	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(configFilename.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	//mm = shapemodels::MorphableModel::loadScmModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\SurreyLowResGuosheng\\NON3448\\ShpVtxModelBin_NON3448.scm", "C:\\Users\\Patrik\\Documents\\GitHub\\featurePoints_SurreyScm.txt");
	//mm = shapemodels::MorphableModel::loadOldBaselH5Model("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\model2012p.h5", "featurePoints_head_newfmt.txt");
	//mm = shapemodels::MorphableModel::loadStatismoModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\2012.2\\head\\model2012_l7_head.h5");
	//shapemodels::MorphableModel morphableModel = shapemodels::MorphableModel::loadScmModel("C:\\Users\\Patrik\\Cloud\\PhD\\MorphModel\\ShpVtxModelBin.scm", "C:\\Users\\Patrik\\Documents\\GitHub\\featurePoints_SurreyScm.txt");
	//shapemodels::MorphableModel morphableModel = shapemodels::MorphableModel::loadScmModel(".\\ShpVtxModelBin.scm", ".\\featurePoints_SurreyScm.txt");
	
	morphablemodel::MorphableModel morphableModel;
	try {
		morphableModel = morphablemodel::MorphableModel::load(pt.get_child("morphableModel"));
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	catch (const std::runtime_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	//render::Mesh cube = render::utils::MeshUtils::createCube();
	//render::Mesh pyramid = render::utils::MeshUtils::createPyramid();
	//render::Mesh plane = render::utils::MeshUtils::createPlane();

	int screenWidth = 640;
	int screenHeight = 480;
	const float aspect = (float)screenWidth/(float)screenHeight;

	//Camera camera(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));
	//Camera camera(Vec3f(0.0f, 0.0f, 3.0f), horizontalAngle*(CV_PI/180.0f), verticalAngle*(CV_PI/180.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));
	Mat moveCameraBack = render::utils::MatrixUtils::createTranslationMatrix(0.0f, 0.0f, -3.0f);
	Mat projection = render::utils::MatrixUtils::createOrthogonalProjectionMatrix(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar);
	SoftwareRenderer r(screenWidth, screenHeight);

	//namedWindow(windowName, WINDOW_AUTOSIZE);
	//setMouseCallback(windowName, winOnMouse);

	vector<int> vertexIds; // Todo: Work with the tlms names everywhere instead of the vertex indices
	/* Surrey full model: */
	vertexIds.push_back(11389); // 177: left-eye-left - right.eye.corner_outer
	vertexIds.push_back(25864/*10930*/); // 181: left-eye-right - right.eye.corner_inner
	vertexIds.push_back(25868); // 614: right-eye-left - left.eye.corner_inner. But we take one a little bit more up
	vertexIds.push_back(11395); // 610: right-eye-right - left.eye.corner_outer
	vertexIds.push_back(398); // mouth-left - right.lips.corner
	vertexIds.push_back(812); // mouth-right - left.lips.corner
	vertexIds.push_back(11140); // bridge of the nose - // should use the new one from the reference, MnFMdl.obj
	vertexIds.push_back(114); // nose-tip - center.nose.tip
	vertexIds.push_back(270); // nasal septum - 
	vertexIds.push_back(3284); // left-alare - right nose ...
	vertexIds.push_back(572); // right-alare - left nose ...
	
	// BFM statismo:
/*	vertexIds.push_back(2348); // left-eye-left - right.eye.corner_outer
	vertexIds.push_back(6216); // left-eye-right - right.eye.corner_inner
	vertexIds.push_back(10344); // right-eye-left - left.eye.corner_inner
	vertexIds.push_back(14216); // right-eye-right - left.eye.corner_outer
	vertexIds.push_back(5392); // mouth-left - right.lips.corner
	vertexIds.push_back(11326); // mouth-right - left.lips.corner
	vertexIds.push_back(8287); // bridge of the nose -
	vertexIds.push_back(8320); // nose-tip - center.nose.tip
	vertexIds.push_back(8332); // nasal septum - (center.nose.attachement_to_philtrum (similar but not defined in the BFM-statismo anyway))
	vertexIds.push_back(6389); // left-alare - (a bit similar to right.nose.wing.tip but don't use it)
	vertexIds.push_back(10001); // right-alare - (a bit similar to left.nose.wing.tip but don't use it)
	*/
	ofstream outputFile;
	//outputFile.open("C:/Users/Patrik/Documents/Github/syndata/data_3dmm_landmarks_random_sd0.5_batch4_600000k.txt");
	outputFile.open("./data_3dmm_landmarks_random_sd0.7_batch6_1000k.txt");
	outputFile << "frontal_rndvtx_x " << "frontal_rndvtx_y " << "frontal_lel_x " << "frontal_lel_y " << "frontal_ler_x " << "frontal_ler_y " << "frontal_rel_x " << "frontal_rel_y " << "frontal_rer_x " << "frontal_rer_y " << "frontal_ml_x " << "frontal_ml_y " << "frontal_mr_x " << "frontal_mr_y " << "frontal_bn_x " << "frontal_bn_y " << "frontal_nt_x " << "frontal_nt_y " << "frontal_ns_x " << "frontal_ns_y " << "frontal_la_x " << "frontal_la_y " << "frontal_ra_x " << "frontal_ra_y " << "pose_rndvtx_x " << "pose_rndvtx_y " << "pose_lel_x " << "pose_lel_y " << "pose_ler_x " << "pose_ler_y " << "pose_rel_x " << "pose_rel_y " << "pose_rer_x " << "pose_rer_y " << "pose_ml_x " << "pose_ml_y " << "pose_mr_x " << "pose_mr_y " << "pose_bn_x " << "pose_bn_y " << "pose_nt_x " << "pose_nt_y " << "pose_ns_x " << "pose_ns_y " << "pose_la_x " << "pose_la_y " << "pose_ra_x " << "pose_ra_y " << "yaw " << "pitch " << "roll " << "rndvtx_id" << std::endl;
	
	int numVertices = morphableModel.getShapeModel().getDataDimension() / 3;
	std::uniform_int_distribution<int> distribution(0, numVertices-1);
	std::mt19937 engine; // Mersenne twister MT19937
	//std::random_device rd;
	//engine.seed(rd());
	engine.seed();
	auto randIntVtx = std::bind(distribution, engine);

	std::uniform_int_distribution<int> distrRandYaw(-55, 55);
	auto randIntYaw = std::bind(distrRandYaw, engine);
	std::uniform_int_distribution<int> distrRandPitch(-40, 40);
	auto randIntPitch = std::bind(distrRandPitch, engine);
	//std::uniform_int_distribution<int> distrRandRoll(-20, 20);
	std::uniform_int_distribution<int> distrRandRoll(0, 0);
	auto randIntRoll = std::bind(distrRandRoll, engine);
	
	/*
	r.resetBuffers();
	r.camera.horizontalAngle = horizontalAngle*(CV_PI/180.0f);
	r.camera.verticalAngle = verticalAngle*(CV_PI/180.0f);
	r.camera.updateFixed(r.camera.eye, r.camera.gaze);
	r.updateViewTransform();
	r.updateProjectionTransform(perspective);
	*/

	int generatedSamples = 0;
	int samplesToGenerate = 600000; //600000;
	int cnt = 0;
	while (generatedSamples < samplesToGenerate) {
		++cnt;
		//r.resetBuffers();
		std::cout << generatedSamples << std::endl;

		render::Mesh newSampleMesh = morphableModel.drawSample(0.7f); // Note: it would suffice to only draw a shape model, but then we can't render it
		render::Mesh::writeObj(newSampleMesh, "C:\\Users\\Patrik\\Documents\\GitHub\\test3.obj");
		
		int randomVertex = randIntVtx();
		int yaw = randIntYaw();
		int pitch = randIntPitch();
		int roll = randIntRoll();

		vector<shared_ptr<Landmark>> pointsToWrite;
		//Mat testImg = r.getImage().clone();

		// 1) Render the randomVertex frontal
		Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f);
		Mat rotPitchX = Mat::eye(4, 4, CV_32FC1);
		Mat rotYawY = Mat::eye(4, 4, CV_32FC1);
		Mat rotRollZ = Mat::eye(4, 4, CV_32FC1);
		Mat modelMatrix = projection * rotYawY * moveCameraBack * rotPitchX * rotRollZ * modelScaling;
		Vec3f res = r.projectVertex(newSampleMesh.vertex[randomVertex].position, modelMatrix);
		auto framebuffers = r.render(newSampleMesh, modelMatrix);
		string name = "randomVertexFrontal";
		pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));
		cv::circle(framebuffers.first, cv::Point(res[0], res[1]), 3, cv::Scalar(0, 0, 255));

		// 2) Render all LMs in frontal pose
		for (const auto& vid : vertexIds) {
			modelMatrix = projection * moveCameraBack * rotYawY * rotPitchX * rotRollZ * modelScaling; // same as before in 1)
			res = r.projectVertex(newSampleMesh.vertex[vid].position, modelMatrix);
			//r.renderLM(newSampleMesh.vertex[vid].position, Scalar(255.0f, 0.0f, 0.0f));
			name = DidLandmarkFormatParser::didToTlmsName(vid);
			pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));
			cv::circle(framebuffers.first, cv::Point(res[0], res[1]), 3, cv::Scalar(255, 0, 128));
		}
		imwrite("out/" + lexical_cast<string>(cnt) + "_front.png", framebuffers.first);

		// 3) Render the randomVertex in pose angle
		rotPitchX = render::utils::MatrixUtils::createRotationMatrixX(pitch * (CV_PI/180.0f));
		rotYawY = render::utils::MatrixUtils::createRotationMatrixY(yaw * (CV_PI/180.0f));
		rotRollZ = render::utils::MatrixUtils::createRotationMatrixZ(roll * (CV_PI/180.0f));
		modelMatrix = projection * moveCameraBack * rotYawY * rotPitchX * rotRollZ * modelScaling;
		auto framebuffersPose = r.render(newSampleMesh, modelMatrix);

		res = r.projectVertex(newSampleMesh.vertex[randomVertex].position, modelMatrix);
		double zBufferValue = framebuffersPose.second.at<double>(static_cast<int>(cvRound(res[1])), static_cast<int>(cvRound(res[0])));
		Point2i centerPixel(floor(res[0]), floor(res[1]));
		int minzx = std::max(0, centerPixel.x - 1);
		int maxzx = std::min(centerPixel.x + 1, framebuffersPose.second.cols - 1);
		int minzy = std::max(0, centerPixel.y - 1);
		int maxzy = std::min(centerPixel.y + 1, framebuffersPose.second.rows - 1);
		bool isVisible = false;
		for (int x = minzx; x < maxzx; ++x) {
			for (int y = minzy; y < maxzy; ++y) {
				double zBufferValue = framebuffersPose.second.at<double>(y, x);
				if (res[2] <= zBufferValue) {
					isVisible = true;
				}
			}
		}
		if (isVisible == false) {
		//if (res[2] > zBufferValue + 0.00004) { // we apply a threshold because our projectVertex is somehow a little bit off, probably because rasterTriangle() works a bit different? (offset, rounding, ...).
			// But caution, this hack depends on the resolution of the z-buffer?
			// not visible
			cv::circle(framebuffersPose.first, cv::Point(res[0], res[1]), 3, cv::Scalar(255, 0, 128));
			imwrite("out/" + lexical_cast<string>(cnt)+"_pose_invis.png", framebuffersPose.first);
			continue;
		}
		cv::circle(framebuffersPose.first, cv::Point(res[0], res[1]), 3, cv::Scalar(255, 0, 128));
		name = "randomVertexPose";
		pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));

		// 4) Render all LMs in pose angle
		for (const auto& vid : vertexIds) {
			modelMatrix = projection * moveCameraBack * rotYawY * rotPitchX * rotRollZ * modelScaling; // same as before in 3)
			res = r.projectVertex(newSampleMesh.vertex[vid].position, modelMatrix);
			name = DidLandmarkFormatParser::didToTlmsName(vid);
			pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));
			cv::circle(framebuffersPose.first, cv::Point(res[0], res[1]), 3, cv::Scalar(128, 0, 255));
		}
		imwrite("out/" + lexical_cast<string>(cnt)+"_pose_vis.png", framebuffersPose.first);
					
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
