/*
 * pose-from-landmarks.cpp
 *
 *  Created on: 25.11.2014
 *      Author: Patrik Huber
 *
 * Example:
 * pose-from-landmarks -c ../../FeatureDetection/fitter/share/configs/default.cfg -i ../../data/iBug_lfpw/testset/image_0001.png -l ../../data/iBug_lfpw/testset/image_0001.pts -t ibug -m ../../FeatureDetection/libImageIO/share/landmarkMappings/ibug2did.txt -o ../../out/fitter/
 *   
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

//#define WIN32_LEAN_AND_MEAN
//#define VC_EXTRALEAN
//#include <windows.h>
//#include <gl/GL.h>

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

#include "boost/math/constants/constants.hpp"

#include "Eigen/Dense"

#include "morphablemodel/MorphableModel.hpp"

#include "fitting/AffineCameraEstimation.hpp"
#include "fitting/OpenCVCameraEstimation.hpp"
#include "fitting/LinearShapeFitting.hpp"

#include "render/SoftwareRenderer.hpp"
#include "render/Camera.hpp"
#include "render/MeshUtils.hpp"
#include "render/matrixutils.hpp"
#include "render/utils.hpp"

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
#include "imageio/MuctLandmarkFormatParser.hpp"
#include "imageio/SimpleModelLandmarkFormatParser.hpp"
#include "imageio/LandmarkMapper.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using render::Mesh;
using cv::Mat;
using cv::Point2f;
using cv::Vec3f;
using cv::Scalar;
using boost::property_tree::ptree;
using boost::filesystem::path;
using boost::lexical_cast;
using std::cout;
using std::endl;
using std::make_shared;
using cv::Point3f;

#include "openglwindow.h"
#include <QtGui/QGuiApplication>
#include <QtGui/QMatrix4x4>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QScreen>
#include <QtCore/qmath.h>
class TriangleWindow : public OpenGLWindow
{
public:
	TriangleWindow();

	void initialize() Q_DECL_OVERRIDE;
	void render() Q_DECL_OVERRIDE;

private:
	GLuint loadShader(GLenum type, const char *source);

	GLuint m_posAttr;
	GLuint m_colAttr;
	GLuint m_matrixUniform;

	QOpenGLShaderProgram *m_program;
	int m_frame;
};

TriangleWindow::TriangleWindow()
	: m_program(0)
	, m_frame(0)
{
}

float radiansToDegrees(float radians) // change to constexpr in VS2014
{
	return radians * static_cast<float>(180 / CV_PI);
};

float degreesToRadians(float degrees) // change to constexpr in VS2014
{
	return degrees * static_cast<float>(CV_PI / 180);
};

// fovy to lrtb: http://nehe.gamedev.net/article/replacement_for_gluperspective/21002/

float cot(float x)
{
	return std::tan(M_PI_2 - x);
};

// fovy is the field of view in degrees, along the y axis
float fovyToFocalLength(float fovy, float height)
{
	// The actual equation is: $ cot(fov/2) = f/(h/2) $
	// equivalent to: $ f = (h/2) * cot(fov/2) $
	// Now I assume that in OpenGL, h = 2 (-1 to 1), so this simplifies to
	// $ f = cot(fov/2) $, which corresponds with http://wiki.delphigl.com/index.php/gluPerspective.
	// It also coincides with Rafaels Formula.

	return (cot(degreesToRadians(fovy) / 2) * (height / 2.0f));
};

// height: window height in Rafael's case. But I think more correct is |top-bottom| in NDC/clip coords, i.e. 2?
float focalLengthToFovy(float focalLength, float height)
{
	return radiansToDegrees(2.0f * std::atan2(height, 2.0f * focalLength)); // both are always positive, so atan() should do it as well?
};

class OpenCVPositWrapper
{
public:
	// TODO: Actually, split the "get the correspondences" to another function, and input two vectors of points here. Same for affine cam esti!
	// img only for debug purposes
	// Todo: Change the returned pair to out-params (refs)
	std::pair<cv::Mat, cv::Mat> estimate(std::vector<imageio::ModelLandmark> imagePoints, cv::Mat img, morphablemodel::MorphableModel morphableModel)
	{
		// Todo: Currently, the optional vertexIds are not used
		// Get the corresponding 3DMM vertices:
		std::vector<Point3f> mmVertices;
		for (auto&& l : imagePoints) {
			Vec3f point;
			try {
				point = morphableModel.getShapeModel().getMeanAtPoint(l.getName());
			}
			catch (std::out_of_range& e) {
				// Todo Log?
				continue;
			}
			mmVertices.emplace_back(cvPoint3D32f(point[0], point[1], point[2]));
		}
		// move all 3D points so that the first one is now (0, 0, 0) (cvPOSIT requirement)
		Point3f originVertex = mmVertices.front();
		std::transform(begin(mmVertices), end(mmVertices), begin(mmVertices), [&originVertex](const Point3f& p) {return (p - originVertex); });

		// Convert to old C-API format for cvPOSIT:
		vector<CvPoint3D32f> modelPoints;
		for (auto&& v : mmVertices) {
			modelPoints.emplace_back(cvPoint3D32f(v.x, v.y, v.z));
		}

		// Create the 2D image points:
		vector<CvPoint2D32f> srcImagePoints;
		for (auto&& p : imagePoints) {
			// re-center the points and store:
			Point2f positCoords = convertToImageCenterOrigin(p.getPoint2D(), img.cols, img.rows);
			srcImagePoints.emplace_back(cvPoint2D32f(positCoords.x, positCoords.y));
		}

		// Create the POSIT object with the model points. Wrap it in a unique_ptr with custom deleter.
		std::unique_ptr<CvPOSITObject, void(*)(CvPOSITObject*)> positObject = std::unique_ptr<CvPOSITObject, void(*)(CvPOSITObject*)>(cvCreatePOSITObject(&modelPoints[0], (int)modelPoints.size()), [](CvPOSITObject *p) { cvReleasePOSITObject(&p); });

		// Estimate the pose:
		Mat rotation_matrix(3, 3, CV_32FC1);
		Mat translation_vector(1, 3, CV_32FC1);
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1.0e-4f);
		cvPOSIT(positObject.get(), &srcImagePoints[0], 1000, criteria, rotation_matrix.ptr<float>(0), translation_vector.ptr<float>(0)); // 1000 = focal length

		// Visualize the points:
		for (auto&& p : imagePoints) {
			cv::circle(img, p.getPoint2D(), 3, { 255, 0, 0 });
		}
		for (auto&& v : mmVertices) {
			Point2f projpoint = project(v, translation_vector, rotation_matrix, 1000.0f);
			projpoint = convertFromImageCenterOrigin(projpoint, img.cols, img.rows);
			cv::circle(img, projpoint, 3, { 0, 0, 255 });
		}

		return std::make_pair(translation_vector, rotation_matrix);
	};

private:
	// Project a point from 3D to 2D using R, t, and a focal length
	// 
	// Caution: The vertex should already be moved to correspond to the
	// correct origin, i.e. the one chosen at the time of running cvPOSIT.
	//
	// Note: Move to render/fitting/utils, where we already have some project() methods?
	Point2f project(Point3f vertex, Mat translation_vector, Mat rotation_matrix, float focalLength) {
		Mat vertex3dproj = rotation_matrix * Mat(vertex) + translation_vector.t();
		Vec3f vertex3dprojvec(vertex3dproj);
		Point2f projpoint;
		if (vertex3dprojvec[2] != 0) {
			projpoint.x = focalLength * vertex3dprojvec[0] / vertex3dprojvec[2];
			projpoint.y = focalLength * vertex3dprojvec[1] / vertex3dprojvec[2];
		}
		return projpoint;
	};

	// POSIT expects the image center to be in the middle of the image (probably
	// because that's where the camera points to), thus we have to move all the points.

	// Convert from image-center-origin to normal [0, w-1] x [0, h-1] coords
	Point2f convertFromImageCenterOrigin(Point2f positCoords, int imgWidth, int imgHeight) {
		float centerX = ((float)(imgWidth - 1)) / 2.0f;
		float centerY = ((float)(imgHeight - 1)) / 2.0f;
		return Point2f(centerX + positCoords.x, centerY - positCoords.y);
	};

	// Convert from normal coordinates to image-center-origin coords
	Point2f convertToImageCenterOrigin(Point2f imgCoords, int imgWidth, int imgHeight) {
		float centerX = ((float)(imgWidth - 1)) / 2.0f;
		float centerY = ((float)(imgHeight - 1)) / 2.0f;
		return Point2f(imgCoords.x - centerX, centerY - imgCoords.y);
	};
};

class OpenCVsolvePnPWrapper
{
public:
	// TODO: Actually, split the "get the correspondences" to another function, and input two vectors of points here. Same for affine cam esti!
	// img only for debug purposes
	// camMatrix = K: intrinsic camera matrix. Focal length, principal point.
	// From the OpenCV doc of solvePnP: "R, t bring points from the model coordinate system to the camera coordinate system."
	// R is a 3x3 rotation matrix whose columns are the directions of the world axes in the camera's reference frame? Not sure if the case in solvePnP.
	// Todo: Change the returned pair to out-params (refs)
	// Returns: [R t; 0 0 0 1]: how to transform points in world coordinates to camera coordinates (rotation followed by translation)
	std::pair<cv::Mat, cv::Mat> estimate(std::vector<imageio::ModelLandmark> landmarks, cv::Mat camMatrix, cv::Mat img, morphablemodel::MorphableModel morphableModel)
	{
		// Todo: Currently, the optional vertexIds are not used
		std::vector<Point2f> imagePoints;
		// Get the corresponding 3DMM vertices:
		std::vector<Point3f> mmVertices;
		for (auto&& l : landmarks) {
			Vec3f point;
			try {
				point = morphableModel.getShapeModel().getMeanAtPoint(l.getName());
			}
			catch (std::out_of_range& e) {
				// Todo Log?
				continue;
			}
			mmVertices.emplace_back(cvPoint3D32f(point[0], point[1], point[2]));
			imagePoints.emplace_back(l.getPosition2D());
		}

		for (auto&& p : imagePoints) {
			cv::circle(img, p, 3, { 255, 0, 0 });
		}

		Mat rvec(3, 1, CV_64FC1);
		Mat tvec(3, 1, CV_64FC1);
		solvePnP(mmVertices, imagePoints, camMatrix, vector<float>(), rvec, tvec, false, CV_EPNP); // CV_ITERATIVE (3pts) | CV_P3P (4pts) | CV_EPNP (4pts)

		// Visualize the points:
		Mat rotation_matrix(3, 3, CV_64FC1);
		Rodrigues(rvec, rotation_matrix);
		rotation_matrix.convertTo(rotation_matrix, CV_32FC1);
		Mat translation_vector = tvec;
		translation_vector.convertTo(translation_vector, CV_32FC1);

		camMatrix.convertTo(camMatrix, CV_32FC1);

		Mat extrinsicCameraMatrix = Mat::zeros(4, 4, CV_32FC1);
		rotation_matrix.copyTo(extrinsicCameraMatrix.rowRange(0, 3).colRange(0, 3));
		translation_vector.copyTo(extrinsicCameraMatrix.rowRange(0, 3).col(3));
		extrinsicCameraMatrix.at<float>(3, 3) = 1;

		Mat intrinsicCameraMatrix = Mat::zeros(4, 4, CV_32FC1);
		camMatrix.copyTo(intrinsicCameraMatrix.rowRange(0, 3).colRange(0, 3));
		intrinsicCameraMatrix.at<float>(3, 3) = 1;

		for (auto&& v : mmVertices) {
			Mat vertex_homo = Mat::ones(4, 1, CV_32FC1);
			Mat(v).copyTo(vertex_homo.rowRange(0, 3));
			Mat v_after_extr = extrinsicCameraMatrix * vertex_homo;
			Mat v_after_intrin = intrinsicCameraMatrix * v_after_extr;

			Point3f v4p_homo(v_after_intrin.rowRange(0, 3));
			Point2f v4p2d_homo(v4p_homo.x / v4p_homo.z, v4p_homo.y / v4p_homo.z); // if != 0

			cv::circle(img, v4p2d_homo, 3, { 255.0, 0.0, 255.0 });
		}

		return std::make_pair(tvec, rvec);
	};

private:
	// Project a point from 3D to 2D using R, t, and a focal length
	// 
	// Caution: The vertex should already be moved to correspond to the
	// correct origin, i.e. the one chosen at the time of running cvPOSIT.
	//
	// Note: Move to render/fitting/utils, where we already have some project() methods?
	// TODO ADJUST FOR SOLVEPNP
	Point2f project(Point3f vertex, Mat translation_vector, Mat rotation_matrix, float focalLength) {
		Mat vertex3dproj = rotation_matrix * Mat(vertex) + translation_vector.t();
		Vec3f vertex3dprojvec(vertex3dproj);
		Point2f projpoint;
		if (vertex3dprojvec[2] != 0) {
			projpoint.x = focalLength * vertex3dprojvec[0] / vertex3dprojvec[2];
			projpoint.y = focalLength * vertex3dprojvec[1] / vertex3dprojvec[2];
		}
		return projpoint;
	};
};

template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(cout, " "));
	return os;
}

cv::Vec3f eulerAnglesFromRotationMatrix(cv::Mat R)
{
	Vec3f eulerAngles;
	eulerAngles[0] = std::atan2(R.at<float>(2, 1), R.at<float>(2, 2)); // r_32, r_33. Theta_x.
	eulerAngles[1] = std::atan2(-R.at<float>(2, 0), std::sqrt(std::pow(R.at<float>(2, 1), 2) + std::pow(R.at<float>(2, 2), 2))); // r_31, sqrt(r_32^2 + r_33^2). Theta_y.
	eulerAngles[2] = std::atan2(R.at<float>(1, 0), R.at<float>(0, 0)); // r_21, r_11. Theta_z.
	return eulerAngles;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	bool useFileList = false;
	bool useImage = false;
	bool useDirectory = false;
	path inputFilename;
	path configFilename;
	path inputLandmarks;
	string landmarkType;
	path landmarkMappings;
	path outputPath;

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
			("landmark-mappings,m", po::value<path>(&landmarkMappings),
				"an optional mapping-file that maps from the input landmarks to landmark identifiers in the model's format")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: pose-from-landmarks [options]" << endl;
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
	
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("morphablemodel").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("render").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("fitting").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));
	appLogger.debug("Using config: " + configFilename.string());

	// We assume the user has given either an image, directory, or a .lst-file
	if (inputFilename.extension().string() == ".lst" || inputFilename.extension().string() == ".txt") { // check for .lst or .txt first
		useFileList = true;
	}
	else if (boost::filesystem::is_directory(inputFilename)) { // check if it's a directory
		useDirectory = true;
	}
	else { // it must be an image
		useImage = true;
	}
	
	// Load the images
	shared_ptr<ImageSource> imageSource;
	if (useFileList == true) {
		appLogger.info("Using file-list as input: " + inputFilename.string());
		shared_ptr<ImageSource> fileListImgSrc; // TODO VS2013 change to unique_ptr, rest below also
		try {
			fileListImgSrc = make_shared<FileListImageSource>(inputFilename.string());
		}
		catch (const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
		imageSource = fileListImgSrc;
	}
	if (useImage == true) {
		appLogger.info("Using input image: " + inputFilename.string());
		shared_ptr<ImageSource> fileImgSrc;
		try {
			fileImgSrc = make_shared<FileImageSource>(inputFilename.string());
		}
		catch (const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
		imageSource = fileImgSrc;
	}
	if (useDirectory == true) {
		appLogger.info("Using input images from directory: " + inputFilename.string());
		try {
			imageSource = make_shared<DirectoryImageSource>(inputFilename.string());
		}
		catch (const std::runtime_error& e) {
			appLogger.error(e.what());
			return EXIT_FAILURE;
		}
	}

	// Load the ground truth
	shared_ptr<LabeledImageSource> labeledImageSource;
	shared_ptr<NamedLandmarkSource> landmarkSource;
	
	shared_ptr<LandmarkFormatParser> landmarkFormatParser;
	string landmarksFileExtension(".txt");
	if(boost::iequals(landmarkType, "ibug")) {
		landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
		landmarksFileExtension = ".pts";
	}
	else if (boost::iequals(landmarkType, "did")) {
		landmarkFormatParser = make_shared<DidLandmarkFormatParser>();
		//landmarksFileExtension = ".did";
		landmarksFileExtension = ".pos";
	}
	else if (boost::iequals(landmarkType, "muct76-opencv")) {
		landmarkFormatParser = make_shared<MuctLandmarkFormatParser>();
		landmarksFileExtension = ".csv";
	}
	else if (boost::iequals(landmarkType, "SimpleModelLandmark")) {
		landmarkFormatParser = make_shared<SimpleModelLandmarkFormatParser>();
		landmarksFileExtension = ".txt";
	}
	else {
		cout << "Error: Invalid ground truth type." << endl;
		return EXIT_FAILURE;
	}
	if (useImage == true) {
		// The user can either specify a filename, or, as in the other input-cases, a directory
		if (boost::filesystem::is_directory(inputLandmarks)) {
			landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, landmarksFileExtension, GatherMethod::ONE_FILE_PER_IMAGE_DIFFERENT_DIRS, vector<path>{ inputLandmarks }), landmarkFormatParser);
		}
		else {
			landmarkSource = make_shared<DefaultNamedLandmarkSource>(vector<path>{ inputLandmarks }, landmarkFormatParser);
		}
	}
	if (useFileList == true) {
		//landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, landmarksFileExtension, GatherMethod::ONE_FILE_PER_IMAGE_DIFFERENT_DIRS, vector<path>{ inputLandmarks }), landmarkFormatParser);
		landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, landmarksFileExtension, GatherMethod::SEPARATE_FOLDERS_RECURSIVE, vector<path>{ inputLandmarks }), landmarkFormatParser);
	}
	if (useDirectory == true) {
		landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, landmarksFileExtension, GatherMethod::ONE_FILE_PER_IMAGE_DIFFERENT_DIRS, vector<path>{ inputLandmarks }), landmarkFormatParser);
	}
	labeledImageSource = make_shared<NamedLabeledImageSource>(imageSource, landmarkSource);
	
	// Read the config file
	ptree config;
	try {
		boost::property_tree::info_parser::read_info(configFilename.string(), config);
	} catch(const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	// Load the Morphable Model
	morphablemodel::MorphableModel morphableModel;
	try {
		morphableModel = morphablemodel::MorphableModel::load(config.get_child("morphableModel"));
	} catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	catch (const std::runtime_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	// Create the output directory if it doesn't exist yet
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	vector<imageio::ModelLandmark> landmarks;
	float lambda = config.get_child("fitting", ptree()).get<float>("lambda", 15.0f);

	//LandmarkMapper landmarkMapper(landmarkMappings);
	LandmarkMapper landmarkMapper;
	if (!landmarkMappings.empty()) {
		// the user has given a landmark mappings file on the console
		landmarkMapper = LandmarkMapper(landmarkMappings);
	} // Ideas for a better solution: A flag in LandmarkMapper, or polymorphism (IdentityLandmarkMapper), or in Mapper, if mapping empty, return input?, or...?

	while (labeledImageSource->next()) {
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + labeledImageSource->getName().string());
		img = labeledImageSource->getImage();

		LandmarkCollection lms = labeledImageSource->getLandmarks();
		if (lms.isEmpty()) {
			appLogger.warn("No landmarks found for this image. Skipping it!");
			continue;
		}
		LandmarkCollection didLms;
		if (!landmarkMappings.empty()) {
			didLms = landmarkMapper.convert(lms);
		}
		else {
			didLms = lms;
		}

		landmarks.clear();
		Mat landmarksImage = img.clone(); // blue rect = the used landmarks
		for (const auto& lm : didLms.getLandmarks()) {
			lm->draw(landmarksImage);
			landmarks.emplace_back(imageio::ModelLandmark(lm->getName(), lm->getPosition2D()));
			cv::rectangle(landmarksImage, cv::Point(cvRound(lm->getX() - 2.0f), cvRound(lm->getY() - 2.0f)), cv::Point(cvRound(lm->getX() + 2.0f), cvRound(lm->getY() + 2.0f)), cv::Scalar(255, 0, 0));
		}


		vector<imageio::ModelLandmark> landmarksClipSpace = fitting::convertAvailableLandmarksToClipSpace(landmarks, morphableModel, img.cols, img.rows);

		OpenCVsolvePnPWrapper pnp;
		// We got K (3x3), R (3x3) and t (3x1)
		
	/*	Mat K = (cv::Mat_<float>(3, 3) << 1500.0f,    0.0f, img.cols / 2.0f, // subtract 1 somewhere or something? see window transform
											 0.0f, 1500.0f, img.rows / 2.0f,
											 0.0f,    0.0f, 1.0f); // K_33 = -1? Doesn't seem to make a difference for solvePnP. OpenCV expect +1 according to doc.
		// This focal length is in "image space" (pixels). I.e. divide by h/w to get it in "NDC"/clip/... space?
		// We specify the image center for the solvePnP camera.
		// Later, in OpenGL, we don't, because OpenGLs NDC / clip coords are unit cube with center in the middle anyway.
*/
		Mat K = (cv::Mat_<float>(3, 3) << 3.4169f, 0.0f, 0.0f,
										  0.0f, 3.4169f, 0.0f,
										  0.0f, 0.0f, 1.0f);

		Mat tpnp, Rpnp;
		std::tie(tpnp, Rpnp) = pnp.estimate(landmarksClipSpace, K, img, morphableModel);

		Mat t;
		tpnp.convertTo(t, CV_32FC1);
		Mat R(3, 3, CV_64FC1);
		Rodrigues(Rpnp, R);
		R.convertTo(R, CV_32FC1);
		
		// Create a K, a projection matrix that works with OpenGL:
		Mat K_opengl = Mat::zeros(4, 4, CV_32FC1);
		K.copyTo(K_opengl.rowRange(0, 3).colRange(0, 3));
		K_opengl.at<float>(0, 2) = 0.0f;
		K_opengl.at<float>(1, 2) = 0.0f; // camera is in the middle of the screen
		K_opengl.at<float>(2, 2) = 0.0f;
		K_opengl.at<float>(3, 2) = -1.0f; // for OpenGL
		// If we use a pixel-valued focal length in 'K' for solvePnP, we have to divide the focal lengths here by w and/or h. The aspect ration shouldn't play a role here so probably divide both by w, and ortho will take care of the aspect later?
		// Not needed when solvePnP works in clip coords.
		//K_opengl.at<float>(0, 0) = K_opengl.at<float>(0, 0) / img.cols;
		//K_opengl.at<float>(1, 1) = K_opengl.at<float>(1, 1) / img.cols;

		// We need to prevent losing Z-depth information:
		// The new third row preserve the ordering of Z-values while mapping -near and -far onto themselves (after normalizing by w, proof left as an exercise). The result is that points between the clipping planes remain between clipping planes after multiplication by Persp
		float nearPlane = 0.1f;
		float farPlane = 4000.0f;
		K_opengl.at<float>(2, 2) = nearPlane + farPlane; // The 'A'. Not sure if n and f should be pos or neg!
		K_opengl.at<float>(2, 3) = nearPlane * farPlane; // The 'B'

		float aspect = static_cast<float>(img.cols) / img.rows;
		Mat ortho = render::matrixutils::createOrthogonalProjectionMatrix(-1.0f * aspect, 1.0f * aspect, -1.0f, 1.0f, nearPlane, farPlane); // if (b, t) are like this, ortho doesn't switch anything. The viewport transform will later switch the y axis.

		Mat opengl_proj = ortho * K_opengl;

		// Model (and view) matrix (view is identity):
		Mat modelTranslation = render::matrixutils::createTranslationMatrix(t.at<float>(0), t.at<float>(1), -t.at<float>(2)); // flip y (opengl <> opencv origin) and z
		Mat modelR4x4 = Mat::zeros(4, 4, CV_32FC1);
		R.copyTo(modelR4x4.rowRange(0, 3).colRange(0, 3));
		modelR4x4.at<float>(3, 3) = 1.0f;

		Mat opengl_modelview = modelTranslation * modelR4x4;

		// Finished! Now rendering:
		Mesh mesh = morphableModel.getMean();
		render::SoftwareRenderer softwareRenderer(img.cols, img.rows);
		softwareRenderer.doBackfaceCulling = false;
		
		softwareRenderer.clearBuffers();
		auto framebuffer = softwareRenderer.render(mesh, opengl_modelview, opengl_proj);
		Mat renderedModel = framebuffer.first.clone(); // we save that later, and the framebuffer gets overwritten
		Mat renderedModelZ = framebuffer.second.clone();

		{ // only for debug purposes:
			cv::Vec3f eulerAngles = eulerAnglesFromRotationMatrix(R);
			float theta_x_deg = radiansToDegrees(eulerAngles[0]); // Pitch. Positive means the subject looks down.
			float theta_y_deg = radiansToDegrees(eulerAngles[1]); // Yaw. Positive means the subject looks left, i.e. we see mre of his right side.
			float theta_z_deg = radiansToDegrees(eulerAngles[2]); // Roll. Positive means the subjects (real) right eye moves downwards.
		}

		// Some general notes:
		// t_z = -1900: Move the model points further into the screen. Equivalent to "the camera is at +500 (world-space) and the points stay".
		//float fovy = focalLengthToFovy(1500.0f, img.rows);
		//float fovy = focalLengthToFovy(3.4169, 2);
		// My rotation matrices are the transposes of those returned by QR. Maybe extrinsic/intrinsic rotation? (i.e. one rotates the camera, one the object)
		// ---
		// Actually now of course R is equivalent to 'createRotationMatrixZ(theta_z) * createRotationMatrixY(theta_y) * createRotationMatrixX(theta_x)', so no need to extract these angles anymore. Just extend from 3x3 to 4x4.
		// solvePnP uses rotation order Z * Y * X * vec
		// ---
		// Actually, YEAH, we REALLY want the opposite! (i.e. don't scale the model). We're interested in the camera parameters, i.e. where's the camera. This way, we can also determine the face size (because we know the model is in milimetres). Only problem in tracking: The face AND the camera might both move, or only the face.
		// ---
		// NOTE: Aah, our window transform kind of "screws it up". It flips our y-axis because
		// OpenGL has the origin bottom-left, but OpenCV has it top-left. That's good.
		// However, as solvePnP directly estimates from the model-space to the "OpenCV-window-space", its
		// R and t are of course flipped as well. However, the ANGLES are the same in all cases.
		// This also explains why my rotation matrices are the transposes of the solvePnP one's (possibly also for POSIT etc.)

		/*
		QGuiApplication app(argc, argv);
		QSurfaceFormat format;
		format.setSamples(16);
		TriangleWindow window;
		window.setFormat(format);
		window.resize(640, 480);
		window.show();
		window.setAnimating(true);
		return app.exec();
		*/
		/*
		OpenCVPositWrapper posit;
		Mat t, R;
		std::tie(t, R) = posit.estimate(landmarks, img, morphableModel);
		// Based off http://nghiaho.com/?page_id=846, http://planning.cs.uiuc.edu/node102.html
		// apply in the order:  in the order theta_X -> theta_Y -> theta_Z
		// == the Tait-Bryan angles (or the xyz convention), not Euler angles
		float theta_x = std::atan2(R.at<float>(2, 1), R.at<float>(2, 2)); // r_32, r_33
		float theta_y = std::atan2(-R.at<float>(2, 0), std::sqrt(std::pow(R.at<float>(2, 1), 2) + std::pow(R.at<float>(2, 2), 2))); // r_31, sqrt(r_32^2 + r_33^2)
		float theta_z = std::atan2(R.at<float>(1, 0), R.at<float>(0, 0)); // r_21, r_11
		
		float theta_x_deg = radiansToDegrees(theta_x);
		float theta_y_deg = radiansToDegrees(theta_y); // Should be yaw
		float theta_z_deg = radiansToDegrees(theta_z);

		// compare angles with fitter affine+decomp: 16, 29, 16. Here: 11, 29, 11.
		Mat mtxR, mtxQ, Qx, Qy, Qz;
		cv::Vec3d eulerAngles = RQDecomp3x3(R, mtxR, mtxQ, Qx, Qy, Qz);
		
		// My matrices are the transposes of those returned by QR. Maybe extrinsic/intrinsic rotation? (i.e. one rotates the camera, one the object)
		Mat myrotMX = render::matrixutils::createRotationMatrixX(theta_x);
		Mat myrotMY = render::matrixutils::createRotationMatrixY(theta_y);
		Mat myrotMZ = render::matrixutils::createRotationMatrixZ(theta_z);
		//cv::Rodrigues()

		// Save the fitting file
		ptree fittingFile;
		fittingFile.put("camera", string("posit"));
		fittingFile.put("camera.t.x", t.at<float>(0));
		fittingFile.put("camera.t.y", t.at<float>(1));
		fittingFile.put("camera.t.z", t.at<float>(2));
		fittingFile.put("camera.yaw", 1.0f); // to get the subjects yaw, we have to flip?
		fittingFile.put("camera.pitch", 1.0f);
		fittingFile.put("camera.roll", 1.0f);

		fittingFile.put("imageWidth", img.cols);
		fittingFile.put("imageHeight", img.rows);

		path fittingFileName = outputPath / labeledImageSource->getName().stem();
		fittingFileName += ".txt";
		boost::property_tree::write_info(fittingFileName.string(), fittingFile);
		*/
		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.");
	}
	return 0;
}

static const char *vertexShaderSource =
"attribute highp vec4 posAttr;\n"
"attribute lowp vec4 colAttr;\n"
"varying lowp vec4 col;\n"
"uniform highp mat4 matrix;\n"
"void main() {\n"
"   col = colAttr;\n"
"   gl_Position = matrix * posAttr;\n"
"}\n";

static const char *fragmentShaderSource =
"varying lowp vec4 col;\n"
"void main() {\n"
"   gl_FragColor = col;\n"
"}\n";
//! [3]

//! [4]
GLuint TriangleWindow::loadShader(GLenum type, const char *source)
{
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, 0);
	glCompileShader(shader);
	return shader;
}

void TriangleWindow::initialize()
{
	m_program = new QOpenGLShaderProgram(this);
	m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
	m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
	m_program->link();
	m_posAttr = m_program->attributeLocation("posAttr");
	m_colAttr = m_program->attributeLocation("colAttr");
	m_matrixUniform = m_program->uniformLocation("matrix");
}
//! [4]

//! [5]
void TriangleWindow::render()
{
	const qreal retinaScale = devicePixelRatio();
	glViewport(0, 0, width() * retinaScale, height() * retinaScale);

	glClear(GL_COLOR_BUFFER_BIT);

	m_program->bind();

	QMatrix4x4 matrix;
	matrix.perspective(60.0f, 4.0f / 3.0f, 0.1f, 100.0f);
	matrix.translate(0, 0, -2);
	matrix.rotate(100.0f * m_frame / screen()->refreshRate(), 0, 1, 0);

	m_program->setUniformValue(m_matrixUniform, matrix);

	GLfloat vertices[] = {
		0.0f, 0.707f,
		-0.5f, -0.5f,
		0.5f, -0.5f
	};

	GLfloat colors[] = {
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f
	};

	glVertexAttribPointer(m_posAttr, 2, GL_FLOAT, GL_FALSE, 0, vertices);
	glVertexAttribPointer(m_colAttr, 3, GL_FLOAT, GL_FALSE, 0, colors);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glDrawArrays(GL_TRIANGLES, 0, 3);

	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);

	m_program->release();

	++m_frame;
}
