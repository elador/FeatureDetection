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
#include "morphablemodel/OpenCVCameraEstimation.hpp"
#include "morphablemodel/AffineCameraEstimation.hpp"
#include "render/Camera.hpp"
#include "render/SoftwareRenderer.hpp"

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

#include "OpenGLWindow.hpp"

#include <QtGui/QGuiApplication>
#include <QtGui/QMatrix4x4>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QScreen>

#include <QtCore/qmath.h>
#undef ERROR // from Qt/OGL something... better solution: Rename the loglevels

#include "render/QOpenGLRenderer.hpp"
#include "render/MeshUtils.hpp"

#include <QPainter>
#include <QImage>
#include <QGLWidget>
#include <QtGui/QOpenGLPaintDevice>

using namespace imageio;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using boost::property_tree::ptree;
using boost::filesystem::path;
using boost::lexical_cast;
using cv::Mat;
using cv::Point2f;
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;

// Returns true if inside the tri or on the border
bool isPointInTriangle(cv::Point2f point, cv::Point2f triV0, cv::Point2f triV1, cv::Point2f triV2) {
	/* See http://www.blackpawn.com/texts/pointinpoly/ */
	// Compute vectors        
	cv::Point2f v0 = triV2 - triV0;
	cv::Point2f v1 = triV1 - triV0;
	cv::Point2f v2 = point - triV0;

	// Compute dot products
	float dot00 = v0.dot(v0);
	float dot01 = v0.dot(v1);
	float dot02 = v0.dot(v2);
	float dot11 = v1.dot(v1);
	float dot12 = v1.dot(v2);

	// Compute barycentric coordinates
	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is in triangle
	return (u >= 0) && (v >= 0) && (u + v < 1);
}

// framebuffer where to extract the texture from
// note: framebuffer should have size of the image (ok not necessarily. What about mobile?) (well it should, to get optimal quality (and everywhere the same quality)?)
cv::Mat extractTexture(render::Mesh mesh, QMatrix4x4 mvpMatrix, int viewportWidth, int viewportHeight, cv::Mat framebuffer) { // Change QMatrix4x4 to cv::Mat so that software-renderer is not dependent on Qt?
	// optional param cv::Mat textureMap = cv::Mat(512, 512, CV_8UC3) ?
	//cv::Mat textureMap(512, 512, inputImage.type());
	cv::Mat textureMap(512, 512, CV_8UC3);

	for (const auto& triangleIndices : mesh.tvi) {

		cv::Point2f srcTri[3];
		cv::Point2f dstTri[3];
		QVector4D vec(mesh.vertex[triangleIndices[0]].position[0], mesh.vertex[triangleIndices[0]].position[1], mesh.vertex[triangleIndices[0]].position[2], 1.0f);
		QVector4D res = mvpMatrix * vec;
		res /= res.w();
		float x_w = (res.x() + 1)*(viewportWidth / 2.0f) + 0.0f; // OpenGL viewport transform (from NDC to viewport) (NDC=clipspace?)
		float y_w = (res.y() + 1)*(viewportHeight / 2.0f) + 0.0f;
		y_w = viewportHeight - y_w; // Qt: Origin top-left. OpenGL: bottom-left.
		srcTri[0] = cv::Point2f(x_w, y_w);

		vec = QVector4D(mesh.vertex[triangleIndices[1]].position[0], mesh.vertex[triangleIndices[1]].position[1], mesh.vertex[triangleIndices[1]].position[2], 1.0f);
		res = mvpMatrix * vec;
		res /= res.w();
		x_w = (res.x() + 1)*(viewportWidth / 2.0f) + 0.0f;
		y_w = (res.y() + 1)*(viewportHeight / 2.0f) + 0.0f;
		y_w = viewportHeight - y_w; // Qt: Origin top-left. OpenGL: bottom-left.
		srcTri[1] = cv::Point2f(x_w, y_w);

		vec = QVector4D(mesh.vertex[triangleIndices[2]].position[0], mesh.vertex[triangleIndices[2]].position[1], mesh.vertex[triangleIndices[2]].position[2], 1.0f);
		res = mvpMatrix * vec;
		res /= res.w();
		x_w = (res.x() + 1)*(viewportWidth / 2.0f) + 0.0f;
		y_w = (res.y() + 1)*(viewportHeight / 2.0f) + 0.0f;
		y_w = viewportHeight - y_w; // Qt: Origin top-left. OpenGL: bottom-left.
		srcTri[2] = cv::Point2f(x_w, y_w);

		// ROI in the source image:
		// Todo: Check if the triangle is on screen. If it's outside, we crash here.
		float src_tri_min_x = std::min(srcTri[0].x, std::min(srcTri[1].x, srcTri[2].x)); // note: might be better to round later (i.e. use the float points for getAffineTransform for a more accurate warping)
		float src_tri_max_x = std::max(srcTri[0].x, std::max(srcTri[1].x, srcTri[2].x));
		float src_tri_min_y = std::min(srcTri[0].y, std::min(srcTri[1].y, srcTri[2].y));
		float src_tri_max_y = std::max(srcTri[0].y, std::max(srcTri[1].y, srcTri[2].y));
		
		Mat inputImageRoi = framebuffer.rowRange(cvFloor(src_tri_min_y), cvCeil(src_tri_max_y)).colRange(cvFloor(src_tri_min_x), cvCeil(src_tri_max_x)); // We round down and up. ROI is possibly larger. But wrong pixels get thrown away later when we check if the point is inside the triangle? Correct?
		srcTri[0] -= Point2f(src_tri_min_x, src_tri_min_y);
		srcTri[1] -= Point2f(src_tri_min_x, src_tri_min_y);
		srcTri[2] -= Point2f(src_tri_min_x, src_tri_min_y); // shift all the points to correspond to the roi

		dstTri[0] = cv::Point2f(textureMap.cols*mesh.vertex[triangleIndices[0]].texcrd[0], textureMap.rows*mesh.vertex[triangleIndices[0]].texcrd[1] - 1.0f);
		dstTri[1] = cv::Point2f(textureMap.cols*mesh.vertex[triangleIndices[1]].texcrd[0], textureMap.rows*mesh.vertex[triangleIndices[1]].texcrd[1] - 1.0f);
		dstTri[2] = cv::Point2f(textureMap.cols*mesh.vertex[triangleIndices[2]].texcrd[0], textureMap.rows*mesh.vertex[triangleIndices[2]].texcrd[1] - 1.0f);

		/// Get the Affine Transform
		cv::Mat warp_mat = getAffineTransform(srcTri, dstTri);

		/// Apply the Affine Transform just found to the src image
		cv::Mat tmpDstBuffer = Mat::zeros(textureMap.rows, textureMap.cols, framebuffer.type()); // I think using the source-size here is not correct. The dst might be larger. We should warp the endpoints and set to max-w/h. No, I think it would be even better to directly warp to the final textureMap size. (so that the last step is only a 1:1 copy)
		warpAffine(inputImageRoi, tmpDstBuffer, warp_mat, tmpDstBuffer.size(), cv::INTER_CUBIC, cv::BORDER_TRANSPARENT); // last row/col is zeros, depends on interpolation method. Maybe because of rounding or interpolation? So it cuts a little. Maybe try to implement by myself?

		// only copy to final img if point is inside the triangle (or on the border)
		for (int x = std::min(dstTri[0].x, std::min(dstTri[1].x, dstTri[2].x)); x < std::max(dstTri[0].x, std::max(dstTri[1].x, dstTri[2].x)); ++x) {
			for (int y = std::min(dstTri[0].y, std::min(dstTri[1].y, dstTri[2].y)); y < std::max(dstTri[0].y, std::max(dstTri[1].y, dstTri[2].y)); ++y) {
				if (isPointInTriangle(cv::Point2f(x, y), dstTri[0], dstTri[1], dstTri[2])) {
					textureMap.at<cv::Vec3b>(y, x) = tmpDstBuffer.at<cv::Vec3b>(y, x);
				}
			}
		}
	}
	return textureMap;
}

class FittingWindow : public OpenGLWindow
{
public:
	FittingWindow(shared_ptr<LabeledImageSource> labeledImageSource, morphablemodel::MorphableModel morphableModel);
	~FittingWindow() {
		delete m_device;
	}

	void initialize();
	void render();
	void fit();

private:
	GLuint loadShader(GLenum type, const char *source);

	GLuint m_posAttr;
	GLuint m_colAttr;
	GLuint m_texAttr;
	GLuint m_texWeightAttr;
	GLuint m_matrixUniform;

	GLuint texture;

	QOpenGLShaderProgram *m_program;
	int m_frame;
	
	QOpenGLPaintDevice *m_device;

	shared_ptr<LabeledImageSource> labeledImageSource; // todo unique_ptr
	morphablemodel::MorphableModel morphableModel;
	bool renderModel = false;
};

FittingWindow::FittingWindow(shared_ptr<LabeledImageSource> labeledImageSource, morphablemodel::MorphableModel morphableModel) : m_program(0)
, m_frame(0)
, m_device(0)
{
	this->labeledImageSource = labeledImageSource;
	this->morphableModel = morphableModel;
}


template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(cout, " "));
	return os;
}

int main(int argc, char *argv[])
//int WinMain(int argc, char *argv[])
//int _WinMain(int argc, char **argv)
//int main(int argc, TCHAR *argv[])
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
	path outputPath;
	float lambdaIn;

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
			("output,o", po::value<path>(&outputPath)->required(),
				"alpha out dir")
			("lambda,d", po::value<float>(&lambdaIn)->default_value(0.01),
				"lambda")
				
		;

		po::positional_options_description p;
		p.add("input", -1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise).run(), vm);
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
	
	Loggers->getLogger("morphablemodel").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
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
		} else if (boost::iequals(landmarkType, "did")) {
			landmarkFormatParser = make_shared<DidLandmarkFormatParser>();
			landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, ".did", GatherMethod::ONE_FILE_PER_IMAGE_DIFFERENT_DIRS, groundtruthDirs), landmarkFormatParser);
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

	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	

	/*string faceDetectionModel("C:\\opencv\\opencv_2.4.8\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"); // sgd: "../models/haarcascade_frontalface_alt2.xml"
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(faceDetectionModel))
	{
		cout << "Error loading face detection model." << endl;
		return EXIT_FAILURE;
	}*/

	//render::QOpenGLRenderer r;

	QGuiApplication app(argc, argv);

	QSurfaceFormat format;
	format.setSamples(16);

	FittingWindow window(labeledImageSource, morphableModel);
	window.setFormat(format);
	window.resize(640, 480);
	window.show();

	window.setAnimating(true);

	return app.exec();
	

	return 0;
}

GLuint FittingWindow::loadShader(GLenum type, const char *source)
{
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, 0);
	glCompileShader(shader);
	return shader;
}

void FittingWindow::initialize()
{

}

void FittingWindow::render()
{
	
	++m_frame;
}

void FittingWindow::fit()
{
	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	morphablemodel::OpenCVCameraEstimation epnpCameraEstimation(morphableModel); // todo: this can all go to only init once
	morphablemodel::AffineCameraEstimation affineCameraEstimation(morphableModel);
	vector<imageio::ModelLandmark> landmarks;
	Logger appLogger = Loggers->getLogger("fitter");

	//while (labeledImageSource->next()) {
		labeledImageSource->next();
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + labeledImageSource->getName().string());
		img = labeledImageSource->getImage();

		/*vector<cv::Rect> detectedFaces;
		faceCascade.detectMultiScale(img, detectedFaces, 1.2, 2, 0, cv::Size(50, 50));
		if (detectedFaces.empty()) {
		continue;
		}
		Mat output = img.clone();
		for (const auto& f : detectedFaces) {
		cv::rectangle(output, f, cv::Scalar(0.0f, 0.0f, 255.0f));
		}*/



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

	//}



}




/*
// Start solvePnP & display (e.g. instead of the affine estimation)
int max_d = std::max(img.rows, img.cols); // should be the focal length? (don't forget the aspect ratio!). TODO Read in Hartley-Zisserman what this is
Mat intrCamMatrixTmp = shapemodels::OpenCVCameraEstimation::createIntrinsicCameraMatrix(max_d, img.cols, img.rows);
Mat extrinsicCameraMatrix = epnpCameraEstimation.estimate(landmarks, intrCamMatrixTmp);
intrCamMatrixTmp.convertTo(intrCamMatrixTmp, CV_32FC1);
//vector<Point2f> projectedPoints;
//projectPoints(modelPoints, rvec, tvec, camMatrix, vector<float>(), projectedPoints); // same result as below
Mat intrinsicCameraMatrix = Mat::zeros(4, 4, CV_32FC1);
Mat intrinsicCameraMatrixMain = intrinsicCameraMatrix(cv::Range(0, 3), cv::Range(0, 3));
intrCamMatrixTmp.copyTo(intrinsicCameraMatrixMain);
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
Mat vertex_projected = intrinsicCameraMatrix * extrinsicCameraMatrix * vertex_homo;
Point3f v4p_homo(vertex_projected(cv::Range(0, 3), cv::Range(0, 1)));
Point2f v4p2d_homo(v4p_homo.x / v4p_homo.z, v4p_homo.y / v4p_homo.z); // if != 0
cv::circle(pnpCamLandmarksProjectionImage, v4p2d_homo, 4.0f, Scalar(0.0f, 255.0f, 0.0f));
}
*/

/*
// render with solvePnP:
r.perspectiveDivision = render::SoftwareRenderer::PerspectiveDivision::Z;
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
*/
/*
//r.setModelTransform(render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f));
*/
