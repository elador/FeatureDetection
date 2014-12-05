/*
 * sdm-pose-from-landmarks.cpp
 *
 *  Created on: 05.12.2014
 *      Author: Patrik Huber
 *
 * Example:
 * sdm-pose-from-landmarks -c ../../FeatureDetection/fitter/share/configs/default.cfg -i ../../data/iBug_lfpw/testset/image_0001.png -l ../../data/iBug_lfpw/testset/image_0001.pts -t ibug -m ../../FeatureDetection/libImageIO/share/landmarkMappings/ibug2did.txt -o ../../out/fitter/
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
#include "boost/archive/text_iarchive.hpp"

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

#include "superviseddescent/superviseddescent.hpp"
#include "fitting/utils.hpp"

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
using namespace render::utils;
using namespace superviseddescent;
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

		v2::SupervisedDescentPoseEstimation<v2::LinearRegressor> sdp;
		{
			std::ifstream poseRegressorFile("../sdm-examples-simple-v2/pose_regressor_11lms.txt");
			boost::archive::text_iarchive ia(poseRegressorFile);
			ia >> sdp;
		}

		//sdp.predict(x0, y, h);

		// Finished! Now rendering:
		Mesh mesh = morphableModel.getMean();
		render::SoftwareRenderer softwareRenderer(img.cols, img.rows);
		softwareRenderer.doBackfaceCulling = false;
		/*
		softwareRenderer.clearBuffers();
		auto framebuffer = softwareRenderer.render(mesh, opengl_modelview, opengl_proj);
		Mat renderedModel = framebuffer.first.clone(); // we save that later, and the framebuffer gets overwritten
		Mat renderedModelZ = framebuffer.second.clone();
		*/
	

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
