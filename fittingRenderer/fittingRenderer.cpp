/*
 * fitter.cpp
 *
 *  Created on: 30.05.2014
 *      Author: Patrik Huber
 *
 * Example:
 * fitter -c ../../FeatureDetection/fitter/share/configs/default.cfg -i ../../data/iBug_lfpw/testset/image_0001.png -l ../../data/iBug_lfpw/testset/image_0001.pts -t ibug -m ../../FeatureDetection/libImageIO/share/landmarkMappings/ibug2did.txt -o ../../out/fitter/
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
			("landmark-mappings,m", po::value<path>(&landmarkMappings)->required(),
				"a mapping-file that maps from the input landmarks to landmark identifiers in the model's format")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
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
	float lambda = 15.0f;

	LandmarkMapper landmarkMapper(landmarkMappings);

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

	// Estimate the shape coefficients:
	// Detector variances: Should not be in pixels. Should be normalised by the IED. Normalise by the image dimensions is not a good idea either, it has nothing to do with it. See comment in fitShapeToLandmarksLinear().
	// Let's just use the hopefully reasonably set default value for now (around 3 pixels)
	vector<float> fittedCoeffs = fitting::fitShapeToLandmarksLinear(morphableModel, affineCam, landmarksClipSpace, lambda);

	// Obtain the full mesh and render it using the estimated camera:
	Mesh mesh = morphableModel.drawSample(fittedCoeffs, vector<float>()); // takes standard-normal (not-normalised) coefficients

	render::SoftwareRenderer softwareRenderer(img.cols, img.rows);
	Mat fullAffineCam = fitting::calculateAffineZDirection(affineCam);
	fullAffineCam.at<float>(2, 3) = fullAffineCam.at<float>(2, 2); // Todo: Find out and document why this is necessary!
	fullAffineCam.at<float>(2, 2) = 1.0f;
	softwareRenderer.doBackfaceCulling = true;
	auto framebuffer = softwareRenderer.render(mesh, fullAffineCam); // hmm, do we have the z-test disabled?
	


	// Write:
	// ptree .fit file: cam-params, model-file, model-params-fn(alphas)
	// alphas
	// texmap

	ptree fittingFile;
	fittingFile.put("camera", string("affine"));
	fittingFile.put("camera.matrix", string("2 5.4 232.22"));

	fittingFile.put("imageWidth", img.cols);
	fittingFile.put("imageHeight", img.rows);

	fittingFile.put("fittingParameters.lambda", lambda);

	fittingFile.put("textureMap", path("C:/bla/texture.png").filename().string());
	
	fittingFile.put("model", config.get_child("morphableModel").get<string>("filename")); // This can throw, but the filename should really exist.
	
	// alphas:
	fittingFile.put("shapeCoefficients", "");
	for (size_t i = 0; i < fittedCoeffs.size(); ++i) {
		fittingFile.put("shapeCoefficients." + std::to_string(i), fittedCoeffs[i]);
	}

	path fittingFileName = outputPath / labeledImageSource->getName().stem();
	fittingFileName += ".txt";
	boost::property_tree::write_info(fittingFileName.string(), fittingFile);

	// Additional optional output, as set in the config file:
	if (config.get_child("output", ptree()).get<bool>("copyInputImage", false)) {
		path outInputImage = outputPath / labeledImageSource->getName().filename();
		cv::imwrite(outInputImage.string(), img);
	}
	if (config.get_child("output", ptree()).get<bool>("landmarksImage", false)) {
		path outLandmarksImage = outputPath / labeledImageSource->getName().stem();
		outLandmarksImage += "_landmarks.png";
		cv::imwrite(outLandmarksImage.string(), framebuffer.first);
	}
	if (config.get_child("output", ptree()).get<bool>("writeObj", false)) {
		path outMesh = outputPath / labeledImageSource->getName().stem();
		outMesh.replace_extension("obj");
		Mesh::writeObj(mesh, outMesh.string());
	}
	if (config.get_child("output", ptree()).get<bool>("renderResult", false)) {
		path outRenderResult = outputPath / labeledImageSource->getName().stem();
		outRenderResult += "_render.png";
		cv::imwrite(outRenderResult.string(), framebuffer.first);
	}
	if (config.get_child("output", ptree()).get<bool>("frontalRendering", false)) {
		throw std::runtime_error("Not implemented yet, please disable.");
	}
	

	end = std::chrono::system_clock::now();
	int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds)+"ms.\n");

	return 0;
}
