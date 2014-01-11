/*
 * fitter.cpp
 *
 *  Created on: 28.12.2013
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

#include "shapemodels/MorphableModel.hpp"
#include "shapemodels/CameraEstimation.hpp"
#include "shapemodels/AffineCameraEstimation.hpp"
#include "render/Camera.hpp"
#include "render/RenderDevicePnP.hpp"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/NamedLabeledImageSource.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/IbugLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using boost::property_tree::ptree;
using boost::filesystem::path;
using boost::lexical_cast;
using cv::Mat;
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;


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
	bool useFileList = false;
	bool useImgs = false;
	bool useDirectory = false;
	bool useLandmarkFiles = false;
	vector<path> inputPaths;
	path inputFilelist;
	path inputDirectory;
	vector<path> inputFilenames;
	path configFilename;
	shared_ptr<ImageSource> imageSource;
	path landmarksDir; // TODO: Make more dynamic wrt landmark format. a) What about the loading-flags (1_Per_Folder etc) we have? b) Expose those flags to cmdline? c) Make a LmSourceLoader and he knows about a LM_TYPE (each corresponds to a Parser/Loader class?)
	string landmarkType;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("config,c", po::value<path>(&configFilename)->required(), 
				"path to a config (.cfg) file")
			("input,i", po::value<vector<path>>(&inputPaths)->required(), 
				"input from one or more files, a directory, or a  .lst/.txt-file containing a list of images")
			("landmarks,l", po::value<path>(&landmarksDir), 
				"load landmark files from the given folder")
			("landmark-type,t", po::value<string>(&landmarkType), 
				"specify the type of landmarks to load: ibug")
		;

		po::positional_options_description p;
		p.add("input", -1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << "Usage: fitter [options]\n";
			cout << desc;
			return EXIT_SUCCESS;
		}
		if (vm.count("landmarks")) {
			useLandmarkFiles = true;
			if (!vm.count("landmark-type")) {
				cout << "You have specified to use landmark files. Please also specify the type of the landmarks to load via --landmark-type or -t." << endl;
				return EXIT_SUCCESS;
			}
		}

	} catch(std::exception& e) {
		cout << e.what() << endl;
		return EXIT_FAILURE;
	}

	loglevel logLevel;
	if(boost::iequals(verboseLevelConsole, "PANIC")) logLevel = loglevel::PANIC;
	else if(boost::iequals(verboseLevelConsole, "ERROR")) logLevel = loglevel::ERROR;
	else if(boost::iequals(verboseLevelConsole, "WARN")) logLevel = loglevel::WARN;
	else if(boost::iequals(verboseLevelConsole, "INFO")) logLevel = loglevel::INFO;
	else if(boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = loglevel::DEBUG;
	else if(boost::iequals(verboseLevelConsole, "TRACE")) logLevel = loglevel::TRACE;
	else {
		cout << "Error: Invalid loglevel." << endl;
		return EXIT_FAILURE;
	}
	
	Loggers->getLogger("shapemodels").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("render").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("fitter").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("fitter");

	appLogger.debug("Verbose level for console output: " + logging::loglevelToString(logLevel));
	appLogger.debug("Using config: " + configFilename.string());

	if (inputPaths.size() > 1) {
		// We assume the user has given several, valid images
		useImgs = true;
		inputFilenames = inputPaths;
	} else if (inputPaths.size() == 1) {
		// We assume the user has given either an image, directory, or a .lst-file
		if (inputPaths[0].extension().string() == ".lst" || inputPaths[0].extension().string() == ".txt") { // check for .lst or .txt first
			useFileList = true;
			inputFilelist = inputPaths.front();
		} else if (boost::filesystem::is_directory(inputPaths[0])) { // check if it's a directory
			useDirectory = true;
			inputDirectory = inputPaths.front();
		} else { // it must be an image
			useImgs = true;
			inputFilenames = inputPaths;
		}
	} else {
		appLogger.error("Please either specify one or several files, a directory, or a .lst-file containing a list of images to run the program!");
		return EXIT_FAILURE;
	}

	if (useFileList==true) {
		appLogger.info("Using file-list as input: " + inputFilelist.string());
		shared_ptr<ImageSource> fileListImgSrc; // TODO VS2013 change to unique_ptr, rest below also
		try {
			fileListImgSrc = make_shared<FileListImageSource>(inputFilelist.string(), "C:\\Users\\Patrik\\Documents\\GitHub\\data\\fddb\\originalPics\\", ".jpg");
		} catch(const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
		imageSource = fileListImgSrc;
	}
	if (useImgs==true) {
		//imageSource = make_shared<FileImageSource>(inputFilenames);
		//imageSource = make_shared<RepeatingFileImageSource>("C:\\Users\\Patrik\\GitHub\\data\\firstrun\\ws_8.png");
		appLogger.info("Using input images: ");
		vector<string> inputFilenamesStrings;	// Hack until we use vector<path> (?)
		for (const auto& fn : inputFilenames) {
			appLogger.info(fn.string());
			inputFilenamesStrings.push_back(fn.string());
		}
		shared_ptr<ImageSource> fileImgSrc;
		try {
			fileImgSrc = make_shared<FileImageSource>(inputFilenamesStrings);
		} catch(const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
		imageSource = fileImgSrc;
	}
	if (useDirectory==true) {
		appLogger.info("Using input images from directory: " + inputDirectory.string());
		try {
			imageSource = make_shared<DirectoryImageSource>(inputDirectory.string());
		} catch(const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
	}
	// Load the ground truth
	// Either a) use if/else for imageSource or labeledImageSource, or b) use an EmptyLandmarkSoure
	shared_ptr<LabeledImageSource> labeledImageSource;
	shared_ptr<NamedLandmarkSource> landmarkSource;
	if (useLandmarkFiles) {
		vector<path> groundtruthDirs; groundtruthDirs.push_back(landmarksDir); // Todo: Make cmdline use a vector<path>
		shared_ptr<LandmarkFormatParser> landmarkFormatParser;
		if(boost::iequals(landmarkType, "lst")) {
			//landmarkFormatParser = make_shared<LstLandmarkFormatParser>();
			//landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, string(), GatherMethod::SEPARATE_FILES, groundtruthDirs), landmarkFormatParser);
		} else if(boost::iequals(landmarkType, "ibug")) {
			landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
			landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, ".pts", GatherMethod::ONE_FILE_PER_IMAGE_SAME_DIR, groundtruthDirs), landmarkFormatParser);
		} else {
			cout << "Error: Invalid ground truth type." << endl;
			return EXIT_FAILURE;
		}
	} else {
		landmarkSource = make_shared<EmptyLandmarkSource>();
	}
	labeledImageSource = make_shared<NamedLabeledImageSource>(imageSource, landmarkSource);
	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(configFilename.string(), pt);
	} catch(const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	shapemodels::MorphableModel morphableModel;
	try {
		morphableModel = shapemodels::MorphableModel::load(pt.get_child("morphableModel"));
	} catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	catch (const std::runtime_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	const string windowName = "win";

	shapemodels::CameraEstimation cameraEstimation(morphableModel);
	shapemodels::AffineCameraEstimation affineCameraEstimation(morphableModel);
	vector<imageio::ModelLandmark> landmarks;

	cv::namedWindow(windowName);
	



	while(labeledImageSource->next()) {
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + labeledImageSource->getName().string());
		img = labeledImageSource->getImage();
		
		LandmarkCollection lms = labeledImageSource->getLandmarks();
		vector<shared_ptr<Landmark>> lmsv = lms.getLandmarks();
		landmarks.clear();
		Mat landmarksImage = img.clone(); // blue rect = the used landmarks
		for (const auto& lm : lmsv) {
			lm->draw(landmarksImage);
			if (lm->getName() == "right.eye.corner_outer" || lm->getName() == "right.eye.corner_inner" || lm->getName() == "left.eye.corner_outer" || lm->getName() == "left.eye.corner_inner" || lm->getName() == "center.nose.tip" || lm->getName() == "right.lips.corner" || lm->getName() == "left.lips.corner") {
				landmarks.emplace_back(imageio::ModelLandmark(lm->getName(), lm->getPosition2D()));
				cv::rectangle(landmarksImage, cv::Point(cvRound(lm->getX() - 2.0f), cvRound(lm->getY() - 2.0f)), cv::Point(cvRound(lm->getX() + 2.0f), cvRound(lm->getY() + 2.0f)), cv::Scalar(255, 0, 0));
			}
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
			Point2f pp({ p2d.at<float>(0, 0), p2d.at<float>(1, 0) });
			cv::circle(affineCamLandmarksProjectionImage, pp, 4.0f, Scalar(0.0f, 255.0f, 0.0f));
		}
		// End Affine est.
		/* affineCam = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0, 
											  0, 1, 0, 0,
											  0, 0, 0, 1); */
		Mat affineCamZ = affineCam.row(0).colRange(0, 3).cross(affineCam.row(1).colRange(0, 3));
		affineCamZ /= cv::norm(affineCamZ, cv::NORM_L2);

		// Replace the third row with the camera-direction (z)
		// Todo: Take care of sign
		Mat affineCamSubMat = affineCam.row(2).colRange(0, 3);
		affineCamZ.copyTo(affineCamSubMat);
		affineCam.at<float>(2, 3) = 0;

		Mat affineCamFull = Mat::zeros(4, 4, CV_32FC1);
		Mat affineCamFullSub = affineCamFull.rowRange(0, 2);
		affineCam.rowRange(0, 2).copyTo(affineCamFullSub);
		affineCamFullSub = affineCamFull.row(2).colRange(0, 3);
		affineCamZ.copyTo(affineCamFullSub);
		affineCamFull.at<float>(2, 3) = 0.0f;
		affineCamFull.at<float>(3, 3) = 1.0f; // 4th row is (0, 0, 0, 1)

		Vec4f p1(-10.0f, -5.0f, -3.0f, 1.0f);
		Vec4f p2( 10.0f, -5.0f, -3.0f, 1.0f);
		Vec4f p3( 10.0f, -5.0f,  0.0f, 1.0f);
		Vec4f p4( 10.0f, -5.0f,  3.0f, 1.0f);
		Vec4f p5( 10.0f, -5.0f, 30.0f, 1.0f);
		Point3f pp1(Mat(affineCam * Mat(p1)));
		float pp1z = Mat(affineCamZ * Mat(p1).rowRange(0, 3)).at<float>(0, 0);
		Point3f pp2(Mat(affineCam * Mat(p2)));
		float pp2z = Mat(affineCamZ * Mat(p2).rowRange(0, 3)).at<float>(0, 0);
		Point3f pp3(Mat(affineCam * Mat(p3)));
		float pp3z = Mat(affineCamZ * Mat(p3).rowRange(0, 3)).at<float>(0, 0);
		Point3f pp4(Mat(affineCam * Mat(p4)));
		float pp4z = Mat(affineCamZ * Mat(p4).rowRange(0, 3)).at<float>(0, 0);
		Point3f pp5(Mat(affineCam * Mat(p5)));
		float pp5z = Mat(affineCamZ * Mat(p5).rowRange(0, 3)).at<float>(0, 0);

		// Estimate the shape coefficients

		// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
		// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
		Mat V_hat_h = Mat::zeros(4 * landmarks.size(), morphableModel.getShapeModel().getNumberOfPrincipalComponents(), CV_32FC1);
		int rowIndex = 0;
		for (const auto& lm : landmarks) {
			Mat basisRows = morphableModel.getShapeModel().getPcaBasis(lm.getName()); // getPcaBasis should return the not-normalized basis I think
			Mat submatrixToReplace = V_hat_h.rowRange(rowIndex, rowIndex + 3); // submatrixToReplace is just a pointer to V_hat_h
			basisRows.copyTo(submatrixToReplace);
			rowIndex += 4; // replace 3 rows and skip the 4th one, it has all zeros
		}
		// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affineCam) is placed on the diagonal:
		Mat P = Mat::zeros(3 * landmarks.size(), 4 * landmarks.size(), CV_32FC1);
		for (int i = 0; i < landmarks.size(); ++i) {
			Mat submatrixToReplace = P.colRange(4*i, (4*i)+4).rowRange(3*i, (3*i)+3);
			//Mat submatrixToReplace2 = P.;
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
			y.at<float>(3*i, 0) = landmarks[i].getX();
			y.at<float>((3*i)+1, 0) = landmarks[i].getY();
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
		float lambda = 0.005f; // The weight of the regularisation
		int numShapePc = morphableModel.getShapeModel().getNumberOfPrincipalComponents();
		Mat AtOmegaA = A.t() * Omega * A;
		Mat AtOmegaAReg = AtOmegaA + lambda * Mat::eye(numShapePc, numShapePc, CV_32FC1);
		Mat AtOmegaARegInv = AtOmegaAReg.inv(/*cv::DECOMP_SVD*/);
		Mat AtOmegatb = A.t() * Omega.t() * b;
		Mat c_s = -AtOmegaARegInv * AtOmegatb;
		vector<float> fittedCoeffs(c_s);

		// End estimate the shape coefficients

		// Start solvePnP & display
		int max_d = std::max(img.rows, img.cols); // should be the focal length? (don't forget the aspect ratio!). TODO Read in Hartley-Zisserman what this is
		//int max_d = 700;
		Mat camMatrix = (cv::Mat_<double>(3, 3) << max_d, 0, img.cols / 2.0,
			0, max_d, img.rows / 2.0,
			0, 0, 1.0);

		std::pair<Mat, Mat> rotTransRodr = cameraEstimation.estimate(landmarks, camMatrix);
		Mat rvec = rotTransRodr.first;
		Mat tvec = rotTransRodr.second;

		Mat rotation_matrix(3, 3, CV_64FC1);
		cv::Rodrigues(rvec, rotation_matrix);
		rotation_matrix.convertTo(rotation_matrix, CV_32FC1);
		Mat translation_vector = tvec;
		translation_vector.convertTo(translation_vector, CV_32FC1);

		camMatrix.convertTo(camMatrix, CV_32FC1);

		//vector<Point2f> projectedPoints;
		//projectPoints(modelPoints, rvec, tvec, camMatrix, vector<float>(), projectedPoints); // same result as below
		Mat extrinsicCameraMatrix = Mat::zeros(4, 4, CV_32FC1);
		Mat extrRot = extrinsicCameraMatrix(cv::Range(0, 3), cv::Range(0, 3));
		rotation_matrix.copyTo(extrRot);
		Mat extrTrans = extrinsicCameraMatrix(cv::Range(0, 3), cv::Range(3, 4));
		translation_vector.copyTo(extrTrans);
		extrinsicCameraMatrix.at<float>(3, 3) = 1;

		Mat intrinsicCameraMatrix = Mat::zeros(4, 4, CV_32FC1);
		Mat intrinsicCameraMatrixMain = intrinsicCameraMatrix(cv::Range(0, 3), cv::Range(0, 3));
		camMatrix.copyTo(intrinsicCameraMatrixMain);
		intrinsicCameraMatrix.at<float>(3, 3) = 1;

		vector<Point3f> points3d;
		for (const auto& landmark : landmarks) {
			points3d.emplace_back(morphableModel.getShapeModel().getMeanAtPoint(landmark.getName()));
		}
		Mat pnpCamLandmarksProjectionImage = landmarksImage.clone();
		for (const auto& v : points3d) {
			Mat vertex(v);
			Mat vertex_homo = Mat::ones(4, 1, CV_32FC1);
			Mat vertex_homo_coords = vertex_homo(cv::Range(0, 3), cv::Range(0, 1));
			vertex.copyTo(vertex_homo_coords);
			Mat v2 = rotation_matrix * vertex;
			Mat v3 = v2 + translation_vector;
			Mat v3_mat = extrinsicCameraMatrix * vertex_homo;

			Mat v4 = camMatrix * v3;
			Mat v4_mat = intrinsicCameraMatrix * v3_mat;

			Point3f v4p(v4);
			Point2f v4p2d(v4p.x / v4p.z, v4p.y / v4p.z); // if != 0
			Point3f v4p_homo(v4_mat(cv::Range(0, 3), cv::Range(0, 1)));
			Point2f v4p2d_homo(v4p_homo.x / v4p_homo.z, v4p_homo.y / v4p_homo.z); // if != 0
			cv::circle(pnpCamLandmarksProjectionImage, v4p2d_homo, 4.0f, Scalar(0.0f, 255.0f, 0.0f));
		}

		std::shared_ptr<render::Mesh> meshToDraw = std::make_shared<render::Mesh>(morphableModel.getMean());
		render::Mesh::writeObj(*meshToDraw.get(), "C:\\Users\\Patrik\\Documents\\GitHub\\test.obj");



		const float aspect = (float)img.cols / (float)img.rows; // 640/480
		render::Camera camera(Vec3f(0.0f, 0.0f, 0.0f), /*horizontalAngle*/0.0f*(CV_PI / 180.0f), /*verticalAngle*/0.0f*(CV_PI / 180.0f), render::Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, /*zNear*/-0.1f, /*zFar*/-100.0f));
		render::RenderDevicePnP r(img.cols, img.rows, camera); // 640, 480
		r.perspectiveDivision = render::RenderDevicePnP::PerspectiveDivision::None;
		r.doClippingInNDC = false;
		r.directToScreenTransform = true;
		r.doWindowTransform = false;
		r.setObjectToScreenTransform(affineCamFull);
		r.draw(meshToDraw, nullptr);
		Mat buff = r.getImage();
		Mat buffA = buff.clone();

		
		//r.setModelTransform(render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f));
		r.perspectiveDivision = render::RenderDevicePnP::PerspectiveDivision::Z;
		r.setObjectToScreenTransform(intrinsicCameraMatrix * extrinsicCameraMatrix);
		r.resetBuffers();
		r.draw(meshToDraw, nullptr);
		Mat buffB = r.getImage();
		Mat buffWithoutAlpha;
		cvtColor(buffB, buffWithoutAlpha, cv::COLOR_BGRA2BGR);
		Mat weighted = img.clone(); // get the right size
		cv::addWeighted(pnpCamLandmarksProjectionImage, 0.2, buffWithoutAlpha, 0.8, 0.0, weighted);
		//return std::make_pair(translation_vector, rotation_matrix);
		//img = weighted;
		Mat buffMean = buffB.clone();
		Mat weightedMean = weighted.clone();

		meshToDraw = std::make_shared<render::Mesh>(morphableModel.drawSample(fittedCoeffs, vector<float>(morphableModel.getColorModel().getNumberOfPrincipalComponents(), 0.0f)));
		render::Mesh::writeObj(*meshToDraw.get(), "C:\\Users\\Patrik\\Documents\\GitHub\\testf.obj");
		r.resetBuffers();
		r.draw(meshToDraw, nullptr);
		buffB = r.getImage();
		cvtColor(buffB, buffWithoutAlpha, cv::COLOR_BGRA2BGR);
		weighted = img.clone(); // get the right size
		cv::addWeighted(pnpCamLandmarksProjectionImage, 0.2, buffWithoutAlpha, 0.8, 0.0, weighted);

		cv::imshow(windowName, img);
		cv::waitKey(5);

		// TODO: REPROJECT THE POINTS FROM THE C_S MODEL HERE AND SEE IF THE LMS REALLY GO FURTHER OUT OR JUST THE REST OF THE MESH


		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");

	}

	return 0;
}
