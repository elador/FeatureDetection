/*
 * sdm-train-v2.cpp
 *
 *  Created on: 15.11.2014
 *      Author: Patrik Huber
 */

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#ifdef WIN32
	#include <SDKDDKVer.h>
#endif

/*	// There's a bug in boost/optional.hpp that prevents us from using the debug-crt with it
	// in debug mode in windows. It works in release mode, but as we need debugging, let's
	// disable the windows-memory debugging for now.
#ifdef WIN32
	#include <crtdbg.h>
#endif

#ifdef _DEBUG
	#ifndef DBG_NEW
		#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
		#define new DBG_NEW
	#endif
#endif  // _DEBUG
*/

#include <memory>
#include <iostream>
#include <cmath>

#include "opencv2/core/core.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/archive/text_oarchive.hpp"

#include "render/matrixutils.hpp"
#include "render/SoftwareRenderer.hpp"
#include "render/Camera.hpp"
#include "render/utils.hpp"
#include "morphablemodel/MorphableModel.hpp"
#include "superviseddescent/poseestimation.hpp" // v2 stuff
#include "fitting/utils.hpp"

#include "logging/LoggerFactory.hpp"

using namespace superviseddescent;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::string;
using cv::Mat;
using boost::filesystem::path;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;

cv::Mat packParameters(fitting::ModelFitting modelFitting)
{
	Mat params(1, 6, CV_32FC1);
	params.at<float>(0) = modelFitting.rotationX;
	params.at<float>(1) = modelFitting.rotationY;
	params.at<float>(2) = modelFitting.rotationZ;
	params.at<float>(3) = modelFitting.tx;
	params.at<float>(4) = modelFitting.ty;
	params.at<float>(5) = modelFitting.tz.get();
	return params;
}

fitting::ModelFitting unpackParameters(cv::Mat parameters)
{
	fitting::ModelFitting modelFitting;
	modelFitting.rotationX = parameters.at<float>(0);
	modelFitting.rotationY = parameters.at<float>(1);
	modelFitting.rotationZ = parameters.at<float>(2);
	modelFitting.tx = parameters.at<float>(3);
	modelFitting.ty = parameters.at<float>(4);
	modelFitting.tz = parameters.at<float>(5);
	return modelFitting;
}

template<typename ForwardIterator, typename T>
void strided_iota(ForwardIterator first, ForwardIterator last, T value, T stride)
{
	while (first != last) {
		*first++ = value;
		value += stride;
	}
}

class h_proj
{
public:
	// additional stuff needed by this specific 'h'
	// M = 3d model, 3d points.
	h_proj(Mat model) : model(model)
	{

	};
	
	// the generic params of 'h', i.e. exampleId etc.
	// Here: R, t. (, f)
	// Returns the transformed 2D coords
	cv::Mat operator()(Mat parameters)
	{
		//project the 3D model points using the current params
		fitting::ModelFitting unrolledParameters = unpackParameters(parameters);
		Mat rotPitchX = render::matrixutils::createRotationMatrixX(render::utils::degreesToRadians(unrolledParameters.rotationX));
		Mat rotYawY = render::matrixutils::createRotationMatrixY(render::utils::degreesToRadians(unrolledParameters.rotationY));
		Mat rotRollZ = render::matrixutils::createRotationMatrixZ(render::utils::degreesToRadians(unrolledParameters.rotationZ));
		Mat translation = render::matrixutils::createTranslationMatrix(unrolledParameters.tx, unrolledParameters.ty, -1900.0f);
		Mat modelMatrix = translation * rotYawY * rotPitchX * rotRollZ;
		const float aspect = static_cast<float>(640) / static_cast<float>(480);
		float fovY = render::utils::focalLengthToFovy(1500.0f, 480);
		Mat projectionMatrix = render::matrixutils::createPerspectiveProjectionMatrix(fovY, aspect, 0.1f, 5000.0f);

		int numLandmarks = model.cols;
		Mat new2dProjections(1, numLandmarks * 2, CV_32FC1);
		for (int lm = 0; lm < numLandmarks; ++lm) {
			Vec3f vtx2d = render::utils::projectVertex(cv::Vec4f(model.col(lm)), projectionMatrix * modelMatrix, 640, 480);
			new2dProjections.at<float>(lm) = vtx2d[0]; // the x coord
			new2dProjections.at<float>(lm + numLandmarks) = vtx2d[1]; // y coord
		}

		// Todo/Note: Write a fuction in libFitting: render(ModelFitting) ?
		return new2dProjections;
	};

	cv::Mat model;
};

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: sdm-examples-simple-v2 [options]" << std::endl;
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
	if(boost::iequals(verboseLevelConsole, "PANIC")) logLevel = LogLevel::Panic;
	else if(boost::iequals(verboseLevelConsole, "ERROR")) logLevel = LogLevel::Error;
	else if(boost::iequals(verboseLevelConsole, "WARN")) logLevel = LogLevel::Warn;
	else if(boost::iequals(verboseLevelConsole, "INFO")) logLevel = LogLevel::Info;
	else if(boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = LogLevel::Debug;
	else if(boost::iequals(verboseLevelConsole, "TRACE")) logLevel = LogLevel::Trace;
	else {
		cout << "Error: Invalid LogLevel." << endl;
		return EXIT_FAILURE;
	}
	
	Loggers->getLogger("superviseddescent").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	// Generate RPY
	path morphableModelConfigFilename(R"(C:\Users\Patrik\Documents\GitHub\FeatureDetection\fitter\share\configs\default.cfg)");
	morphablemodel::MorphableModel morphableModel;
	try {
		boost::property_tree::ptree pt;
		boost::property_tree::info_parser::read_info(morphableModelConfigFilename.string(), pt);
		morphableModel = morphablemodel::MorphableModel::load(pt.get_child("morphableModel"));
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	render::Mesh meanMesh = morphableModel.getMean();

	vector<int> vertexIds;
	// 17 iBug points, valid in both Surrey models:
	vertexIds.push_back(225);
	vertexIds.push_back(233);
	vertexIds.push_back(157);
	vertexIds.push_back(590);
	vertexIds.push_back(666);
	vertexIds.push_back(658);
	vertexIds.push_back(114);
	vertexIds.push_back(270);
	vertexIds.push_back(177);
	vertexIds.push_back(181);
	vertexIds.push_back(614);
	vertexIds.push_back(610);
	vertexIds.push_back(398);
	vertexIds.push_back(329);
	vertexIds.push_back(812);
	vertexIds.push_back(411);

	int screenWidth = 1000;
	int screenHeight = 1000;
	const float aspect = static_cast<float>(screenWidth) / static_cast<float>(screenHeight);
	render::SoftwareRenderer renderer(screenWidth, screenHeight);

	const auto yawSeed = std::random_device()();
	std::mt19937 engineYaw; engineYaw.seed(yawSeed);
	std::uniform_real_distribution<> distrRandYaw(-30, 30);
	auto randRealYaw = std::bind(distrRandYaw, engineYaw);

	const auto pitchSeed = std::random_device()();
	std::mt19937 enginePitch; enginePitch.seed(pitchSeed);
	std::uniform_real_distribution<> distrRandPitch(-30, 30);
	auto randRealPitch = std::bind(distrRandPitch, enginePitch);

	const auto rollSeed = std::random_device()();
	std::mt19937 engineRoll; engineRoll.seed(rollSeed);
	std::uniform_real_distribution<> distrRandRoll(-30, 30);
	auto randRealRoll = std::bind(distrRandRoll, engineRoll);

	const auto txSeed = std::random_device()();
	std::mt19937 engineTx; engineTx.seed(txSeed);
	std::uniform_real_distribution<> distrRandTx(-125, 125); // millimetres, in model space. before: +-400
	auto randRealTx = std::bind(distrRandTx, engineTx);

	const auto tySeed = std::random_device()();
	std::mt19937 engineTy; engineTy.seed(tySeed);
	std::uniform_real_distribution<> distrRandTy(-125, 125);
	auto randRealTy = std::bind(distrRandTy, engineTy);

	const auto tzSeed = std::random_device()();
	std::mt19937 engineTz; engineTz.seed(tzSeed);
	std::uniform_real_distribution<> distrRandTz(-500, 500);
	auto randRealTz = std::bind(distrRandTz, engineTz);

	// Try both and compare: rnd tx, ty (, tz) and with alignment (t, maybe scale?) to eyes
	using fitting::ModelFitting;
	vector<ModelFitting> fittings;
	vector<vector<cv::Vec2f>> landmarks;

	int numSamples = 15000;
	int numTr = 10000;
	for (int i = 0; i < numSamples; ++i) {
		// Generate a random pose:
		ModelFitting fitting;
		Mat modelMatrix;
		{
			// Might be better to also use 'h' here?
			auto yaw = randRealYaw();
			auto pitch = randRealPitch();
			auto roll = randRealRoll();
			using render::utils::degreesToRadians;
			Mat rotPitchX = render::matrixutils::createRotationMatrixX(degreesToRadians(pitch));
			Mat rotYawY = render::matrixutils::createRotationMatrixY(degreesToRadians(yaw));
			Mat rotRollZ = render::matrixutils::createRotationMatrixZ(degreesToRadians(roll));

			auto tx = randRealTx();
			auto ty = randRealTy();
			auto tz = -1200.0f + randRealTz();
			// Todo: Hmm, in testing, we want to initialise, but then refine z as well
			Mat translation = render::matrixutils::createTranslationMatrix(tx, ty, tz); // move the model 2.0 metres away from the camera (plus tz)

			modelMatrix = translation * rotYawY * rotPitchX * rotRollZ;
			fitting = ModelFitting(pitch, yaw, roll, tx, ty, tz, {}, {}, 1800.0f);
		}
		// Random tx, ty, tz, f, scale (camera-scale)
		float fovY = render::utils::focalLengthToFovy(1800.0f, screenHeight); //
		Mat projectionMatrix = render::matrixutils::createPerspectiveProjectionMatrix(fovY, aspect, 1.0f, 5000.0f);

		// Test:
		/*
		using render::utils::degreesToRadians;
		Mat rotPitchX = render::matrixutils::createRotationMatrixX(degreesToRadians(0));
		Mat rotYawY = render::matrixutils::createRotationMatrixY(degreesToRadians(0));
		Mat rotRollZ = render::matrixutils::createRotationMatrixZ(degreesToRadians(0));
		auto tz = -700.0f; // with f = 1800 ==> ied ~145
		//auto tz = -1700.0f; // with f = 1800 ==> ied 68
		Mat translation = render::matrixutils::createTranslationMatrix(125, 0, tz);
		modelMatrix = translation * rotYawY * rotPitchX * rotRollZ;

		renderer.clearBuffers();
		Mat colorbuffer, depthbuffer;
		std::tie(colorbuffer, depthbuffer) = renderer.render(meanMesh, modelMatrix, projectionMatrix);
		*/

		// Project all LMs to 2D:
		vector<cv::Vec2f> lms;
		for (auto&& id : vertexIds) {
			Vec3f vtx2d = render::utils::projectVertex(meanMesh.vertex[id].position, projectionMatrix * modelMatrix, screenWidth, screenHeight);
			vtx2d = (vtx2d - Vec3f(500.0f, 500.0f)) / 1800.0f/*=f*/; // New: normalise the image coordinates of the projection
			lms.emplace_back(cv::Vec2f(vtx2d[0], vtx2d[1]));
		}
	
		fittings.emplace_back(fitting);
		landmarks.emplace_back(lms);
	}

	// Our data:
	// Mat(numTr, numLms*2); (the 2D proj)
	// vector<ModelFitting> bzw Mat(numTr, numParamsInUnrolledVec).
	
	// The landmark projections:
	auto numLandmarks = vertexIds.size();
	Mat y_tr(fittings.size(), numLandmarks * 2, CV_32FC1);
	{
		for (int r = 0; r < y_tr.rows; ++r) {
			for (int lm = 0; lm < vertexIds.size(); ++lm) {
				y_tr.at<float>(r, lm) = landmarks[r][lm][0]; // the x coord
				y_tr.at<float>(r, lm + numLandmarks) = landmarks[r][lm][1]; // y coord
			}
		}
	}

	// The parameters:
	Mat x_tr(fittings.size(), 6, CV_32FC1); // 6 params atm: rx, ry, rz, tx, ty, tz
	{
		for (int r = 0; r < x_tr.rows; ++r) {
			Mat parameterRow = packParameters(fittings[r]);
			parameterRow.copyTo(x_tr.row(r));
		}
	}

	Mat x0 = Mat::zeros(fittings.size(), 6, CV_32FC1); // fixed initialisation of the parameters, all zero, except tz
	x0.col(5) -= 1200.0f;

	// The reduced 3D model only consisting of the vertices we're using to learn the pose:
	// ==> Todo this should be saved with the model! Including the vertexIds!
	Mat reducedModel(4, vertexIds.size(), CV_32FC1); // each vertex is a Vec4f (homogenous coords)
	for (int i = 0; i < vertexIds.size(); ++i) {
		Mat vertexCoords = Mat(meanMesh.vertex[vertexIds[i]].position);
		vertexCoords.copyTo(reducedModel.col(i));
	}
	
	// Split the training data into training and test:
	Mat y_ts = y_tr.rowRange(numTr, y_tr.rows);
	y_tr = y_tr.rowRange(0, numTr);
	Mat x_ts_gt = x_tr.rowRange(numTr, x_tr.rows);
	x_tr = x_tr.rowRange(0, numTr);
	Mat x0_ts = x0.rowRange(numTr, x0.rows);
	x0 = x0.rowRange(0, numTr);

	v2::Regulariser reg(v2::Regulariser::RegularisationType::MatrixNorm, 0.1f);
	vector<v2::LinearRegressor> regressors;
	regressors.emplace_back(v2::LinearRegressor(reg));
	regressors.emplace_back(v2::LinearRegressor(reg));
	regressors.emplace_back(v2::LinearRegressor(reg));
	regressors.emplace_back(v2::LinearRegressor(reg));
	regressors.emplace_back(v2::LinearRegressor(reg));

	v2::SupervisedDescentPoseEstimation<v2::LinearRegressor> sdp(regressors);
	sdp.h.model = reducedModel;
	sdp.modelName = "Surrey"; // morphableModel.get...?
	sdp.modelVertexIds = vertexIds;
	
	auto normalisedLeastSquaresResidual = [](const Mat& prediction, const Mat& groundtruth) {
		return cv::norm(prediction, groundtruth, cv::NORM_L2) / cv::norm(groundtruth, cv::NORM_L2);
	};

	auto onTrainCallback = [&](const cv::Mat& currentPredictions) {
		appLogger.info(std::to_string(normalisedLeastSquaresResidual(currentPredictions, x_tr)));
	};

	sdp.train(x_tr, y_tr, x0, onTrainCallback);
	appLogger.info("Finished training.");
	
	// Test the trained model:
	//Mat y_ts(numValuesTest, 1, CV_32FC1);
	//Mat x_ts_gt(numValuesTest, 1, CV_32FC1);
	//Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // We also just start in the center.
	auto onTestCallback = [&](const cv::Mat& currentPredictions) {
		appLogger.info(std::to_string(normalisedLeastSquaresResidual(currentPredictions, x_ts_gt)));
	};
	cv::Mat res = sdp.test(y_ts, x0_ts, onTestCallback);
	appLogger.info("Result of last regressor: " + std::to_string(normalisedLeastSquaresResidual(res, x_ts_gt)));
	
	// Save the trained model to the filesystem:
	// Todo: Encapsulate in a class PoseRegressor, with all the necessary stuff (e.g. which landmarks etc.)
	std::ofstream learnedModelFile("pose_regressor_ibug_17lms_tr10k.txt");
	boost::archive::text_oarchive oa(learnedModelFile);
	oa << sdp;

	appLogger.info("Saved learned regressor model. Exiting...");

	return 0;
}
