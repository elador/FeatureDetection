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

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
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
			("config,c", po::value<path>(&morphableModelConfigFilename)->required(),
				"path to a config file that specifies which Morphable Model to load")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: generate-sample [options]" << endl;
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

	Loggers->getLogger("generate-sample").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("generate-sample");
	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));


	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(morphableModelConfigFilename.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	
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

	int screenWidth = 640;
	int screenHeight = 480;
	const float aspect = (float)screenWidth/(float)screenHeight;

	//Camera camera(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));
	//Camera camera(Vec3f(0.0f, 0.0f, 3.0f), horizontalAngle*(CV_PI/180.0f), verticalAngle*(CV_PI/180.0f), Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar));
	Mat moveCameraBack = render::utils::MatrixUtils::createTranslationMatrix(0.0f, 0.0f, -3.0f);
	float zNear = 0.01f; // maybe check again what makes sense in our case and with our matrices. I think atm we comply with Sherley? But maybe we should rather comply with OpenGL?
	float zFar = 100.0f;
	Mat projection = render::utils::MatrixUtils::createOrthogonalProjectionMatrix(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar);
	SoftwareRenderer r(screenWidth, screenHeight);

	ofstream outputFile;
	outputFile.open("./data_3dmm_landmarks_random_sd0.7_batch6_1000k.txt");
	outputFile << "frontal_rndvtx_x " << "frontal_rndvtx_y " << "frontal_lel_x " << "frontal_lel_y " << "frontal_ler_x " << "frontal_ler_y " << "frontal_rel_x " << "frontal_rel_y " << "frontal_rer_x " << "frontal_rer_y " << "frontal_ml_x " << "frontal_ml_y " << "frontal_mr_x " << "frontal_mr_y " << "frontal_bn_x " << "frontal_bn_y " << "frontal_nt_x " << "frontal_nt_y " << "frontal_ns_x " << "frontal_ns_y " << "frontal_la_x " << "frontal_la_y " << "frontal_ra_x " << "frontal_ra_y " << "pose_rndvtx_x " << "pose_rndvtx_y " << "pose_lel_x " << "pose_lel_y " << "pose_ler_x " << "pose_ler_y " << "pose_rel_x " << "pose_rel_y " << "pose_rer_x " << "pose_rer_y " << "pose_ml_x " << "pose_ml_y " << "pose_mr_x " << "pose_mr_y " << "pose_bn_x " << "pose_bn_y " << "pose_nt_x " << "pose_nt_y " << "pose_ns_x " << "pose_ns_y " << "pose_la_x " << "pose_la_y " << "pose_ra_x " << "pose_ra_y " << "yaw " << "pitch " << "roll " << "rndvtx_id" << std::endl;
	
	std::mt19937 engine; // Mersenne twister MT19937
	//std::random_device rd;
	//engine.seed(rd());
	engine.seed();
		//r.resetBuffers();
	
	render::Mesh newSampleMesh = morphableModel.drawSample(0.7f); // Note: it would suffice to only draw a shape model, but then we can't render it
	render::Mesh::writeObj(newSampleMesh, "C:\\Users\\Patrik\\Documents\\GitHub\\test3.obj");
		
	int yaw = 40;
	int pitch = 10;
	int roll = 10;

	// 3) Render the randomVertex in pose angle
	Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(1.0f / 140.0f, 1.0f / 140.0f, 1.0f / 140.0f);
	Mat rotPitchX = render::utils::MatrixUtils::createRotationMatrixX(pitch * (CV_PI/180.0f));
	Mat rotYawY = render::utils::MatrixUtils::createRotationMatrixY(yaw * (CV_PI / 180.0f));
	Mat rotRollZ = render::utils::MatrixUtils::createRotationMatrixZ(roll * (CV_PI / 180.0f));
	Mat modelMatrix = projection * moveCameraBack * rotYawY * rotPitchX * rotRollZ * modelScaling;
	auto framebuffersPose = r.render(newSampleMesh, modelMatrix);

	//res = r.projectVertex(newSampleMesh.vertex[randomVertex].position, modelMatrix);
	// the vertex might not be visible, check it (has to coincide with the full rendered image)
	//pointsToWrite.push_back(make_shared<ModelLandmark>(name, res));

	outputFile.close();

	return EXIT_SUCCESS;
}
