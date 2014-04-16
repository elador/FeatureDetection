/*
 * fitter.cpp
 *
 *  Created on: 16.04.2014
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

#include <chrono>
#include <memory>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/lexical_cast.hpp"

#include "morphablemodel/MorphableModel.hpp"
#include "morphablemodel/AffineCameraEstimation.hpp"
#include "morphablemodel/OpenCVCameraEstimation.hpp"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/NamedLabeledImageSource.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/IbugLandmarkFormatParser.hpp"
#include "imageio/DidLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using boost::property_tree::ptree;
using boost::filesystem::path;
using boost::lexical_cast;
using cv::Mat;
using cv::Point2f;
using cv::Vec3f;
using cv::Scalar;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;

template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(cout, " "));
	return os;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path inputFilename;
	path configFilename;
	path inputLandmarks;
	string landmarkType;
	path outputPath(".");

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("config,c", po::value<path>(&configFilename)->required(), 
				"path to a config (.cfg) file")
			("input,i", po::value<path>(&inputFilename)->required(),
				"input filename")
			("landmarks,l", po::value<path>(&inputLandmarks)->required(),
				"input landmarks")
			("landmark-type,t", po::value<string>(&landmarkType)->required(),
				"specify the type of landmarks: ibug")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: fitter [options]\n";
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
	
	Loggers->getLogger("morphablemodel").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("render").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("fitter").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("fitter");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));
	appLogger.debug("Using config: " + configFilename.string());

	// Load the image
	shared_ptr<ImageSource> imageSource;
	try {
		imageSource = make_shared<FileImageSource>(inputFilename.string());
	} catch(const std::runtime_error& e) {
		appLogger.error(e.what());
		return EXIT_FAILURE;
	}

	// Load the ground truth
	shared_ptr<LabeledImageSource> labeledImageSource;
	shared_ptr<NamedLandmarkSource> landmarkSource;
	
	shared_ptr<LandmarkFormatParser> landmarkFormatParser;
	if(boost::iequals(landmarkType, "ibug")) {
		landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
		landmarkSource = make_shared<DefaultNamedLandmarkSource>(vector<path>{inputLandmarks}, landmarkFormatParser);
	} else if (boost::iequals(landmarkType, "did")) {
		landmarkFormatParser = make_shared<DidLandmarkFormatParser>();
		landmarkSource = make_shared<DefaultNamedLandmarkSource>(vector<path>{inputLandmarks}, landmarkFormatParser);
	} else {
		cout << "Error: Invalid ground truth type." << endl;
		return EXIT_FAILURE;
	}
	labeledImageSource = make_shared<NamedLabeledImageSource>(imageSource, landmarkSource);
	
	// Load the config file
	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(configFilename.string(), pt);
	} catch(const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	// Load the Morphable Model
	morphablemodel::MorphableModel morphableModel;
	try {
		morphableModel = morphablemodel::MorphableModel::load(pt.get_child("morphableModel"));
	} catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	catch (const std::runtime_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	
	//const string windowName = "win";

	// Create the output directory if it doesn't exist yet
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	

	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	morphablemodel::OpenCVCameraEstimation epnpCameraEstimation(morphableModel); // todo: this can all go to only init once
	morphablemodel::AffineCameraEstimation affineCameraEstimation(morphableModel);
	vector<imageio::ModelLandmark> landmarks;


	labeledImageSource->next();
	start = std::chrono::system_clock::now();
	appLogger.info("Starting to process " + labeledImageSource->getName().string());
	img = labeledImageSource->getImage();

	LandmarkCollection lms = labeledImageSource->getLandmarks();
	vector<shared_ptr<Landmark>> lmsv = lms.getLandmarks();
	landmarks.clear();
	Mat landmarksImage = img.clone(); // blue rect = the used landmarks
	for (const auto& lm : lmsv) {
		lm->draw(landmarksImage);
		//if (lm->getName() == "right.eye.corner_outer" || lm->getName() == "right.eye.corner_inner" || lm->getName() == "left.eye.corner_outer" || lm->getName() == "left.eye.corner_inner" || lm->getName() == "center.nose.tip" || lm->getName() == "right.lips.corner" || lm->getName() == "left.lips.corner") {
		landmarks.emplace_back(imageio::ModelLandmark(lm->getName(), lm->getPosition2D()));
		cv::rectangle(landmarksImage, cv::Point(cvRound(lm->getX() - 2.0f), cvRound(lm->getY() - 2.0f)), cv::Point(cvRound(lm->getX() + 2.0f), cvRound(lm->getY() + 2.0f)), cv::Scalar(255, 0, 0));
		//}
	}

	// Start affine camera estimation (Aldrian paper)
	Mat affineCamLandmarksProjectionImage = landmarksImage.clone(); // the affine LMs are currently not used (don't know how to render without z-vals)
	Mat affineCam = affineCameraEstimation.estimate(landmarks);
	for (const auto& lm : landmarks) {
		Vec3f tmp = morphableModel.getShapeModel().getMeanAtPoint(lm.getName());
		Mat p(4, 1, CV_32FC1);
		p.at<float>(0, 0) = tmp[0];
		p.at<float>(1, 0) = tmp[1];
		p.at<float>(2, 0) = tmp[2];
		p.at<float>(3, 0) = 1;
		Mat p2d = affineCam * p;
		Point2f pp(p2d.at<float>(0, 0), p2d.at<float>(1, 0)); // Todo: check
		cv::circle(affineCamLandmarksProjectionImage, pp, 4.0f, Scalar(0.0f, 255.0f, 0.0f));
	}
	// End Affine est.

	// Estimate the shape coefficients

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	Mat V_hat_h = Mat::zeros(4 * landmarks.size(), morphableModel.getShapeModel().getNumberOfPrincipalComponents(), CV_32FC1);
	int rowIndex = 0;
	for (const auto& lm : landmarks) {
		Mat basisRows = morphableModel.getShapeModel().getPcaBasis(lm.getName()); // getPcaBasis should return the not-normalized basis I think
		basisRows.copyTo(V_hat_h.rowRange(rowIndex, rowIndex + 3));
		rowIndex += 4; // replace 3 rows and skip the 4th one, it has all zeros
	}
	// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affineCam) is placed on the diagonal:
	Mat P = Mat::zeros(3 * landmarks.size(), 4 * landmarks.size(), CV_32FC1);
	for (int i = 0; i < landmarks.size(); ++i) {
		Mat submatrixToReplace = P.colRange(4 * i, (4 * i) + 4).rowRange(3 * i, (3 * i) + 3);
		affineCam.copyTo(submatrixToReplace);
	}
	// The variances: We set the 3D and 2D variances to one static value for now. $sigma^2_2D = sqrt(1) + sqrt(3)^2 = 4$
	float sigma_2D = std::sqrt(4);
	Mat Sigma = Mat::zeros(3 * landmarks.size(), 3 * landmarks.size(), CV_32FC1);
	for (int i = 0; i < 3 * landmarks.size(); ++i) {
		Sigma.at<float>(i, i) = 1.0f / sigma_2D;
	}
	Mat Omega = Sigma.t() * Sigma;
	// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
	Mat y = Mat::ones(3 * landmarks.size(), 1, CV_32FC1);
	for (int i = 0; i < landmarks.size(); ++i) {
		y.at<float>(3 * i, 0) = landmarks[i].getX();
		y.at<float>((3 * i) + 1, 0) = landmarks[i].getY();
		// the position (3*i)+2 stays 1 (homogeneous coordinate)
	}
	// The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
	Mat v_bar = Mat::ones(4 * landmarks.size(), 1, CV_32FC1);
	for (int i = 0; i < landmarks.size(); ++i) {
		Vec3f modelMean = morphableModel.getShapeModel().getMeanAtPoint(landmarks[i].getName());
		v_bar.at<float>(4 * i, 0) = modelMean[0];
		v_bar.at<float>((4 * i) + 1, 0) = modelMean[1];
		v_bar.at<float>((4 * i) + 2, 0) = modelMean[2];
		// the position (4*i)+3 stays 1 (homogeneous coordinate)
	}

	// Bring into standard regularised quadratic form with diagonal distance matrix Omega
	Mat A = P * V_hat_h;
	Mat b = P * v_bar - y;
	//Mat c_s; // The x, we solve for this! (the variance-normalized shape parameter vector, $c_s = [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$
	float lambda = 0.1f; // lambdaIn; //0.01f; // The weight of the regularisation
	int numShapePc = morphableModel.getShapeModel().getNumberOfPrincipalComponents();
	Mat AtOmegaA = A.t() * Omega * A;
	Mat AtOmegaAReg = AtOmegaA + lambda * Mat::eye(numShapePc, numShapePc, CV_32FC1);
	Mat AtOmegaARegInv = AtOmegaAReg.inv(/*cv::DECOMP_SVD*/);
	Mat AtOmegatb = A.t() * Omega.t() * b;
	Mat c_s = -AtOmegaARegInv * AtOmegatb;
	vector<float> fittedCoeffs(c_s);

	// End estimate the shape coefficients

	//std::shared_ptr<render::Mesh> meanMesh = std::make_shared<render::Mesh>(morphableModel.getMean());
	//render::Mesh::writeObj(*meanMesh.get(), "C:\\Users\\Patrik\\Documents\\GitHub\\mean.obj");

	//const float aspect = (float)img.cols / (float)img.rows; // 640/480
	//render::Camera camera(Vec3f(0.0f, 0.0f, 0.0f), /*horizontalAngle*/0.0f*(CV_PI / 180.0f), /*verticalAngle*/0.0f*(CV_PI / 180.0f), render::Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, /*zNear*/-0.1f, /*zFar*/-100.0f));
	/*render::SoftwareRenderer r(img.cols, img.rows, camera); // 640, 480
	r.perspectiveDivision = render::SoftwareRenderer::PerspectiveDivision::None;
	r.doClippingInNDC = false;
	r.directToScreenTransform = true;
	r.doWindowTransform = false;
	r.setObjectToScreenTransform(shapemodels::AffineCameraEstimation::calculateFullMatrix(affineCam));
	r.draw(meshToDraw, nullptr);
	Mat buff = r.getImage();*/

	/*
	std::ofstream myfile;
	path coeffsFilename = outputPath / labeledImageSource->getName().stem();
	myfile.open(coeffsFilename.string() + ".txt");
	for (int i = 0; i < fittedCoeffs.size(); ++i) {
	myfile << fittedCoeffs[i] * std::sqrt(morphableModel.getShapeModel().getEigenvalue(i)) << std::endl;

	}
	myfile.close();
	*/

	//std::shared_ptr<render::Mesh> meshToDraw = std::make_shared<render::Mesh>(morphableModel.drawSample(fittedCoeffs, vector<float>(morphableModel.getColorModel().getNumberOfPrincipalComponents(), 0.0f)));
	//render::Mesh::writeObj(*meshToDraw.get(), "C:\\Users\\Patrik\\Documents\\GitHub\\fittedMesh.obj");

	//r.resetBuffers();
	//r.draw(meshToDraw, nullptr);
	// TODO: REPROJECT THE POINTS FROM THE C_S MODEL HERE AND SEE IF THE LMS REALLY GO FURTHER OUT OR JUST THE REST OF THE MESH

	//cv::imshow(windowName, img);
	//cv::waitKey(5);


	end = std::chrono::system_clock::now();
	int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds)+"ms.\n");


	return 0;
}
