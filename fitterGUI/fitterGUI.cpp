/*
 * fitterGUI.cpp
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

#include "Eigen/Dense"

#include "morphablemodel/MorphableModel.hpp"

#include "fitting/AffineCameraEstimation.hpp"
#include "fitting/OpenCVCameraEstimation.hpp"
#include "fitting/LinearShapeFitting.hpp"

#include "render/SoftwareRenderer.hpp"
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

// alpha = 4th channel. 0 = fully transparent, 255 = not transparent.
// input: backgroundImage needs to be 8UC3, overlayImage 8UC4
// overlayFactor: the alpha gets multiplied with this. make less transparent.
// returns: img with same type as backgroundImage.
cv::Mat alphaBlend(cv::Mat backgroundImage, cv::Mat overlayImage, float overlayFactor = 1.0f) {
	if (backgroundImage.rows != overlayImage.rows || backgroundImage.cols != overlayImage.cols)	{
		// both images must have the same dimensions.
		return cv::Mat();
	}
	if (overlayImage.channels() != 4) {
		// overlayImage must have 4 channels
		return cv::Mat();
	}
	// check if format is RGBA or BGRA, i.e. that alpha is the 4th channel?
	Mat outputImage(backgroundImage.rows, backgroundImage.cols, backgroundImage.type());
	for (int y = 0; y < outputImage.rows; ++y) { // todo: check which loop should be the outer, i.e. which one is faster
		for (int x = 0; x < outputImage.cols; ++x) {
			cv::Vec3b overlayValues(overlayImage.at<cv::Vec4b>(y, x)[0], overlayImage.at<cv::Vec4b>(y, x)[1], overlayImage.at<cv::Vec4b>(y, x)[2]);
			float alpha = static_cast<float>(overlayImage.at<cv::Vec4b>(y, x)[3]) / 255.0f;
			alpha *= overlayFactor;
			outputImage.at<cv::Vec3b>(y, x) = (1.0f - alpha) * backgroundImage.at<cv::Vec3b>(y, x) + alpha * overlayValues;
		}
	}
	return outputImage;
}

float lambda = 1.0f;
int lambda_slider = 10;
int lambda_slider_max = 1000;
bool renderNew = true;

void on_trackbar(int, void*)
{
	//lambda = static_cast<float>(lambda_slider) / static_cast<float>(lambda_slider_max);
	lambda = static_cast<float>(lambda_slider) / 10.0f;
	renderNew = true;
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
	
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("morphablemodel").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("render").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("fitting").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("fitterGUI").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("fitterGUI");

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
	
	const string windowName = "win";
	cv::namedWindow(windowName);
	cv::createTrackbar("Lambda", windowName, &lambda_slider, lambda_slider_max, on_trackbar);

	// Create the output directory if it doesn't exist yet
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	vector<imageio::ModelLandmark> landmarks;

	LandmarkMapper landmarkMapper(path("C:\\Users\\Patrik\\Documents\\GitHub\\FeatureDetection\\libImageIO\\share\\landmarkMappings\\ibug2did.txt"));

	labeledImageSource->next();
	start = std::chrono::system_clock::now();
	appLogger.info("Starting to process " + labeledImageSource->getName().string());
	img = labeledImageSource->getImage();

	LandmarkCollection lms = labeledImageSource->getLandmarks();
	LandmarkCollection didLms = landmarkMapper.convert(lms);
	landmarks.clear();
	Mat landmarksImage = img.clone(); // blue rect = the used landmarks
	for (const auto& lm : didLms.getLandmarks()) {
		lm->draw(landmarksImage);
		landmarks.emplace_back(imageio::ModelLandmark(lm->getName(), lm->getPosition2D()));
		cv::rectangle(landmarksImage, cv::Point(cvRound(lm->getX() - 2.0f), cvRound(lm->getY() - 2.0f)), cv::Point(cvRound(lm->getX() + 2.0f), cvRound(lm->getY() + 2.0f)), cv::Scalar(255, 0, 0));
	}

	// Start affine camera estimation (Aldrian paper)
	Mat affineCamLandmarksProjectionImage = landmarksImage.clone(); // the affine LMs are currently not used (don't know how to render without z-vals)
	
	// Convert the landmarks to clip-space
	vector<imageio::ModelLandmark> landmarksClipSpace;
	for (const auto& lm : landmarks) {
		cv::Vec2f clipCoords = render::utils::screenToClipSpace(lm.getPosition2D(), img.cols, img.rows);
		imageio::ModelLandmark lmcs(lm.getName(), Vec3f(clipCoords[0], clipCoords[1], 0.0f), lm.isVisible());
		landmarksClipSpace.push_back(lmcs);
	}
	
	Mat affineCam = fitting::estimateAffineCamera(landmarksClipSpace, morphableModel);

	// Render the mean-face landmarks projected using the estimated camera:
	for (const auto& lm : landmarks) {
		Vec3f modelPoint = morphableModel.getShapeModel().getMeanAtPoint(lm.getName());
		cv::Vec2f screenPoint = fitting::projectAffine(modelPoint, affineCam, img.cols, img.rows);
		cv::circle(affineCamLandmarksProjectionImage, Point2f(screenPoint), 4.0f, Scalar(0.0f, 255.0f, 0.0f));
	}

	Mat blendedImg;
	while (true)
	{
	if (renderNew)
	{
	
	start = std::chrono::system_clock::now();
	appLogger.info("Starting to process " + labeledImageSource->getName().string());

	// Estimate the shape coefficients:
	// Detector variances: Should not be in pixels. Should be normalised by the IED. Normalise by the image dimensions is not a good idea either, it has nothing to do with it. See comment in fitShapeToLandmarksLinear().
	// Let's just use the hopefully reasonably set default value for now (around 3 pixels)
	vector<float> fittedCoeffs = fitting::fitShapeToLandmarksLinear(morphableModel, affineCam, landmarksClipSpace, lambda);

	Mesh mesh = morphableModel.drawSample(fittedCoeffs, vector<float>()); // takes standard-normal (not-normalised) coefficients
	//Mesh mesh = morphableModel.getMean();
	render::SoftwareRenderer swr(img.cols, img.rows);
	float aspect = (float)img.cols / float(img.rows);
	Mat ortho = render::utils::MatrixUtils::createOrthogonalProjectionMatrix(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 0.01f, 100.0f);
	Mat model = render::utils::MatrixUtils::createScalingMatrix(1.0f / 140.0f, 1.0f / 140.0f, 1.0f / 140.0f);
	Mat cam = render::utils::MatrixUtils::createTranslationMatrix(0.0f, 0.0f, -2.0f);
	Mat rot = render::utils::MatrixUtils::createRotationMatrixY(0.78f);
	//Mat mytransf = ortho * cam * model;
	//auto fb = swr.render(mesh, ortho * cam * rot * model);
	Mat fullAffineCam = fitting::calculateAffineZDirection(affineCam);
	fullAffineCam.at<float>(2, 3) = fullAffineCam.at<float>(2, 2); // Todo: Find out and document why this is necessary!
	fullAffineCam.at<float>(2, 2) = 1.0f;
	swr.doBackfaceCulling = true;
	auto fb = swr.render(mesh, fullAffineCam); // hmm, do we have the z-test disabled?
	//Mesh::writeObj(mesh, "C:/Users/Patrik/Documents/GitHub/out/m_1.0.obj");


	//std::shared_ptr<render::Mesh> meanMesh = std::make_shared<render::Mesh>(morphableModel.getMean());
	//render::Mesh::writeObj(*meanMesh.get(), "C:\\Users\\Patrik\\Documents\\GitHub\\mean.obj");

	//std::shared_ptr<render::Mesh> meshToDraw = std::make_shared<render::Mesh>(morphableModel.drawSample(fittedCoeffs, vector<float>(morphableModel.getColorModel().getNumberOfPrincipalComponents(), 0.0f)));
	//render::Mesh::writeObj(*meshToDraw.get(), "C:\\Users\\Patrik\\Documents\\GitHub\\fittedMesh.obj");

	// TODO: REPROJECT THE POINTS FROM THE C_S MODEL HERE AND SEE IF THE LMS REALLY GO FURTHER OUT OR JUST THE REST OF THE MESH

	end = std::chrono::system_clock::now();
	int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds)+"ms.");

	blendedImg = alphaBlend(affineCamLandmarksProjectionImage, fb.first, 0.9f);
	//cv::addWeighted(landmarksImage, 0.5, fb.first, 0.5, 0.0, blendedImg);
	renderNew = false;
	} // end if renderNew

	cv::imshow(windowName, blendedImg);
	char key = cv::waitKey(30);
	if (key == 'i') {
		lambda += 0.03f;
		renderNew = true;
	}
	if (key == 'j') {
		lambda -= 0.03f;
		renderNew = true;
	}
	if (key == 'k') {
		lambda = 0.0f;
		renderNew = true;
	}

	appLogger.info("Lambda: " + lexical_cast<string>(lambda));
	}

	return 0;
}
