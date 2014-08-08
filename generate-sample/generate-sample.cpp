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
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/optional.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <functional>

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using namespace render;
using cv::Mat;
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
	float yawAngle, pitchAngle, rollAngle;
	boost::optional<float> shapeSdev, colorSdev;
	// w, h, l (list of which LMs to write out), n = numSamples
	
	try {
		float shapeSdevIn, colorSdevIn;
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO", "show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("model,m", po::value<path>(&morphableModelConfigFilename)->required(),
				"path to a config file that specifies which Morphable Model to load")
			("yaw,y", po::value<float>(&yawAngle)->default_value(0.0f),
				"yaw angle in degrees. The rotations will be applied in the order roll, then pitch, then yaw.")
			("pitch,p", po::value<float>(&pitchAngle)->default_value(0.0f),
				"pitch angle in degrees")
			("roll,r", po::value<float>(&rollAngle)->default_value(0.0f),
				"roll angle in degrees")
			("shape-sdev,s", po::value<float>(&shapeSdevIn),
				"optional parameter for the standard deviation of the normal distribution where the shape model coefficients are drawn from. If omitted, the mean is used.")
			("color-sdev,c", po::value<float>(&colorSdevIn),
				"optional parameter for the standard deviation of the normal distribution where the color model coefficients are drawn from. If omitted, the mean is used.")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: generate-sample [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
		if (vm.count("shape-sdev")) {
			shapeSdev = shapeSdevIn;
		}
		if (vm.count("color-sdev")) {
			colorSdev = colorSdevIn;
		}
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
	const float aspect = static_cast<float>(screenWidth)/static_cast<float>(screenHeight);

	Mat moveCameraBack = render::utils::MatrixUtils::createTranslationMatrix(0.0f, 0.0f, -3.0f);
	float zNear = 0.01f; // maybe check again what makes sense in our case and with our matrices. I think atm we comply with Shirley? But maybe we should rather comply with OpenGL?
	float zFar = 100.0f;
	Mat projection = render::utils::MatrixUtils::createOrthogonalProjectionMatrix(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, zNear, zFar);
	SoftwareRenderer r(screenWidth, screenHeight);

	std::mt19937 engine; // Mersenne twister MT19937
	//std::random_device rd;
	//engine.seed(rd());
	engine.seed();

	vector<float> alphas;
	vector<float> betas;

	if (shapeSdev) {
		std::normal_distribution<float> distribution(0.0f, shapeSdev.get()); // this constructor takes the stddev
		alphas.resize(morphableModel.getShapeModel().getNumberOfPrincipalComponents());
		for (auto&& a : alphas) {
			a = distribution(engine);
		}
	}
	if (colorSdev) {
		std::normal_distribution<float> distribution(0.0f, colorSdev.get()); // this constructor takes the stddev
		betas.resize(morphableModel.getColorModel().getNumberOfPrincipalComponents());
		for (auto&& b : betas) {
			b = distribution(engine);
		}
	}
	
	render::Mesh sampleMesh = morphableModel.drawSample(alphas, betas); // if one of the two vectors is empty, it uses getMean()

	Mat modelScaling = render::utils::MatrixUtils::createScalingMatrix(1.0f / 140.0f, 1.0f / 140.0f, 1.0f / 140.0f);
	Mat rotPitchX = render::utils::MatrixUtils::createRotationMatrixX(pitchAngle * static_cast<float>(CV_PI / 180.0));
	Mat rotYawY = render::utils::MatrixUtils::createRotationMatrixY(yawAngle * static_cast<float>(CV_PI / 180.0));
	Mat rotRollZ = render::utils::MatrixUtils::createRotationMatrixZ(rollAngle * static_cast<float>(CV_PI / 180.0));
	Mat modelMatrix = projection * moveCameraBack * rotYawY * rotPitchX * rotRollZ * modelScaling;
	
	Mat framebuffer, depthbuffer;
	r.clearBuffers(); // technically not necessary as long as we render only 1 sample
	std::tie(framebuffer, depthbuffer) = r.render(sampleMesh, modelMatrix);

	render::Mesh::writeObj(sampleMesh, "output.obj");
	cv::imwrite("output.png", framebuffer);

	appLogger.info("Wrote an output.png and output.obj file.");

	return EXIT_SUCCESS;
}
