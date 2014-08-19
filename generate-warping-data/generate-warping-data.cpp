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
#include "render/matrixutils.hpp"
#include "render/SoftwareRenderer.hpp"
#include "render/Camera.hpp"
#include "render/utils.hpp"
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

using namespace imageio;
using namespace render;
namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using cv::Mat;
using boost::lexical_cast;
using boost::property_tree::ptree;
using boost::filesystem::path;
using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::make_shared;

template<class T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v)
{
    copy(v.begin(), v.end(), std::ostream_iterator<T>(cout, " "));
	return os;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(3759128);
	#endif

	string verboseLevelConsole;
	path configFilename;
	
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO", "show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("model,m", po::value<path>(&configFilename)->required(),
				"path to a config file that specifies which Morphable Model to load")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: generate-warping-data [options]" << endl;
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

	Loggers->getLogger("render").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("morphablemodel").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("generate-warping-data").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("generate-warping-data");
	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(configFilename.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	//mm = shapemodels::MorphableModel::loadOldBaselH5Model("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\model2012p.h5", "featurePoints_head_newfmt.txt");
	//mm = shapemodels::MorphableModel::loadStatismoModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\2012.2\\head\\model2012_l7_head.h5");
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

	int screenWidth = 640;
	int screenHeight = 480;
	const float aspect = (float)screenWidth/(float)screenHeight;
	// Create a new camera at (0, 0, 100) looking down the -z axis:
	Camera camera(Vec3f(0.0f, 0.0f, 100.0f), Vec3f(0.0f, 0.0f, -1.0f), Frustum(-150.0f*aspect, 150.0f*aspect, -150.0f, 150.0f, 1.0f, 500.0f));
	SoftwareRenderer renderer(screenWidth, screenHeight);
	renderer.doBackfaceCulling = true;

	vector<int> vertexIds;
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

	// Create the two output files, write the header-line to it:
	std::stringstream headerLine;
	headerLine << "frontal_rndvtx_x " << "frontal_rndvtx_y " << "frontal_lel_x " << "frontal_lel_y " << "frontal_ler_x " << "frontal_ler_y " << "frontal_rel_x " << "frontal_rel_y " << "frontal_rer_x " << "frontal_rer_y " << "frontal_ml_x " << "frontal_ml_y " << "frontal_mr_x " << "frontal_mr_y " << "frontal_bn_x " << "frontal_bn_y " << "frontal_nt_x " << "frontal_nt_y " << "frontal_ns_x " << "frontal_ns_y " << "frontal_la_x " << "frontal_la_y " << "frontal_ra_x " << "frontal_ra_y " << "pose_rndvtx_x " << "pose_rndvtx_y " << "pose_lel_x " << "pose_lel_y " << "pose_ler_x " << "pose_ler_y " << "pose_rel_x " << "pose_rel_y " << "pose_rer_x " << "pose_rer_y " << "pose_ml_x " << "pose_ml_y " << "pose_mr_x " << "pose_mr_y " << "pose_bn_x " << "pose_bn_y " << "pose_nt_x " << "pose_nt_y " << "pose_ns_x " << "pose_ns_y " << "pose_la_x " << "pose_la_y " << "pose_ra_x " << "pose_ra_y " << "yaw " << "pitch " << "roll " << "rndvtx_id" << std::endl;

	std::ofstream outputFileVisible;
	outputFileVisible.open("./data_3dmm_landmarks_random_sd0.5_batch7_1000k_test.txt");
	outputFileVisible << headerLine.str();
	
	std::ofstream outputFileInvisible;
	outputFileInvisible.open("./data_3dmm_landmarks_random_sd0.5_batch7_1000k_test_invisible.txt");
	outputFileInvisible << headerLine.str();

	int numVertices = morphableModel.getShapeModel().getDataDimension() / 3;
	std::uniform_int_distribution<int> distribution(0, numVertices-1);
	std::mt19937 engine; // Mersenne twister MT19937
	//std::random_device rd;
	//engine.seed(rd());
	engine.seed();
	auto randIntVtx = std::bind(distribution, engine);

	std::uniform_real_distribution<> distrRandYaw(-55, 55);
	auto randRealYaw = std::bind(distrRandYaw, engine);
	std::uniform_real_distribution<> distrRandPitch(-40, 40);
	auto randRealPitch = std::bind(distrRandPitch, engine);
	std::uniform_real_distribution<> distrRandRoll(0, 0); // old: -20, 20
	auto randRealRoll = std::bind(distrRandRoll, engine);
	
	int generatedSamples = 0;
	int samplesToGenerate = 1000000;
	int cnt = 0;
	while (generatedSamples < samplesToGenerate) {
		++cnt;
		std::cout << generatedSamples << std::endl;

		render::Mesh newSampleMesh = morphableModel.drawSample(0.5f); // Note: it would suffice to only draw a shape model, but then we can't render it
		//render::Mesh::writeObj(newSampleMesh, "sample.obj");
		
		int randomVertex = randIntVtx();
		auto yaw = randRealYaw();
		auto pitch = randRealPitch();
		auto roll = randRealRoll();

		vector<shared_ptr<Landmark>> pointsToWrite;

		Mat rotPitchX;
		Mat rotYawY;
		Mat rotRollZ;

		// 1) Render the randomVertex frontal
		Mat modelMatrix = Mat::eye(4, 4, CV_32FC1);
		Mat colorbuffer, depthbuffer;
		Vec3f res = render::utils::projectVertex(newSampleMesh.vertex[randomVertex].position, camera.getProjectionMatrix() * camera.getViewMatrix() * modelMatrix, screenWidth, screenHeight);
		renderer.clearBuffers();
		std::tie(colorbuffer, depthbuffer) = renderer.render(newSampleMesh, camera.getViewMatrix() * modelMatrix, camera.getProjectionMatrix());
		string name = "rndVtxFrontal_" + std::to_string(randomVertex);
		pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));
		cv::circle(colorbuffer, cv::Point(res[0], res[1]), 3, cv::Scalar(0, 0, 255, 255));

		// If the random vertex is invisible in the frontal pose, we discard the sample immediately:
		double zBufferValue = depthbuffer.at<double>(static_cast<int>(floor(res[1])), static_cast<int>(floor(res[0])));
		bool isVisibleInFrontal = false;
		if (res[2] - 0.0020 <= zBufferValue) { // it's "in front or on" the zBuffer value, i.e. visible
			isVisibleInFrontal = true;
		}
		if (!isVisibleInFrontal) {
			continue; // draw a new sample
		}

		// 2) Render all LMs in frontal pose
		for (const auto& vid : vertexIds) {
			modelMatrix = Mat::eye(4, 4, CV_32FC1); // same as before in 1)
			res = render::utils::projectVertex(newSampleMesh.vertex[vid].position, camera.getProjectionMatrix() * camera.getViewMatrix() * modelMatrix, screenWidth, screenHeight);
			//r.renderLM(newSampleMesh.vertex[vid].position, Scalar(255.0f, 0.0f, 0.0f));
			pointsToWrite.push_back(make_shared<ModelLandmark>(std::to_string(vid), res));
			//cv::circle(colorbuffer, cv::Point(res[0], res[1]), 3, cv::Scalar(255, 0, 128, 255));
		}
		//imwrite("out/" + lexical_cast<string>(cnt) + "_front.png", colorbuffer);

		// 3) Render the randomVertex in pose angle
		rotPitchX = render::matrixutils::createRotationMatrixX(static_cast<float>(pitch * (CV_PI / 180.0)));
		rotYawY = render::matrixutils::createRotationMatrixY(static_cast<float>(yaw * (CV_PI / 180.0)));
		rotRollZ = render::matrixutils::createRotationMatrixZ(static_cast<float>(roll * (CV_PI / 180.0)));
		modelMatrix = rotYawY * rotPitchX * rotRollZ;
		renderer.clearBuffers();
		std::tie(colorbuffer, depthbuffer) = renderer.render(newSampleMesh, camera.getViewMatrix() * modelMatrix, camera.getProjectionMatrix());

		res = render::utils::projectVertex(newSampleMesh.vertex[randomVertex].position, camera.getProjectionMatrix() * camera.getViewMatrix() * modelMatrix, screenWidth, screenHeight);
		zBufferValue = depthbuffer.at<double>(static_cast<int>(floor(res[1])), static_cast<int>(floor(res[0])));
		cv::Point2i centerPixel(std::floor(res[0]), std::floor(res[1]));
		//cv::circle(colorbuffer, centerPixel, 3, cv::Scalar(255.0f, 0.0f, 0.0f, 255.0f));

		// Method 1 (simple):
		// Says a lot of time the vertex is invisible, when it is in fact visible.
		// Trying with adding a threshold: 0.0005. I think we should set it to 0.0020. We could even try 0.0025 or even 0.003.
		// Beware it depends on the camera settings!
		bool isVisible = false;
		if (res[2] - 0.0020 <= zBufferValue) { // it's "in front or on" the zBuffer value, i.e. visible
			isVisible = true;
		}

		/*
		// Method 2 (neighbours):
		// get the indices to loop around the neighbours of centerPixel
		int minzx = std::max(0, centerPixel.x - 1);
		int maxzx = std::min(centerPixel.x + 1, depthbuffer.cols - 1);
		int minzy = std::max(0, centerPixel.y - 1);
		int maxzy = std::min(centerPixel.y + 1, depthbuffer.rows - 1);
		for (int x = minzx; x < maxzx; ++x) {
			for (int y = minzy; y < maxzy; ++y) {
				double zBufferValue = depthbuffer.at<double>(y, x);
				if (res[2] <= zBufferValue) { // swap sign?
					isVisible = true;
				}
			}
		}
		*/

		name = "rndVtxPose_" + std::to_string(randomVertex);
		pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));

		// 4) Render all LMs in pose angle
		for (const auto& vid : vertexIds) {
			modelMatrix = rotYawY * rotPitchX * rotRollZ; // same as before in 3)
			res = render::utils::projectVertex(newSampleMesh.vertex[vid].position, camera.getProjectionMatrix() * camera.getViewMatrix() * modelMatrix, screenWidth, screenHeight);
			pointsToWrite.push_back(make_shared<ModelLandmark>(std::to_string(vid), res));
			//cv::circle(colorbuffer, cv::Point(res[0], res[1]), 3, cv::Scalar(128, 0, 255, 255));
		}
		//imwrite("out/" + lexical_cast<string>(cnt) + "_pose_vis.png", colorbuffer);
					
		// 4) Write one row to either the visible or invisible vertices file:
		if (isVisible) {
			for (const auto& lm : pointsToWrite) {
				//lm->draw(colorbuffer);
				outputFileVisible << lm->getX() << " " << lm->getY() << " ";
			}
			outputFileVisible << yaw << " " << pitch << " " << roll << " " << randomVertex << std::endl;
		}
		else {
			for (const auto& lm : pointsToWrite) {
				//lm->draw(colorbuffer);
				outputFileInvisible << lm->getX() << " " << lm->getY() << " ";
			}
			outputFileInvisible << yaw << " " << pitch << " " << roll << " " << randomVertex << std::endl;
		}

		++generatedSamples;
	}

	outputFileVisible.close();
	outputFileInvisible.close();

	return 0;
}
