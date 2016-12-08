
#ifdef WIN32
	#include <SDKDDKVer.h>
#endif



#include <chrono>
#include <memory>
#include <iostream>
#include <stdlib.h>

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

#include "superviseddescent/SdmLandmarkModel.hpp"
#include "superviseddescent/LandmarkBasedSupervisedDescentTraining.hpp" // Todo: move the free functions there somewhere else and then remove this include

#include "Eigen/Dense"

#include "morphablemodel/MorphableModel.hpp"

#include "fitting/AffineCameraEstimation.hpp"
#include "fitting/OpenCVCameraEstimation.hpp"
#include "fitting/LinearShapeFitting.hpp"

#include "render/SoftwareRenderer.hpp"
#include "render/MeshUtils.hpp"
#include "render/utils.hpp"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/CameraImageSource.hpp"
#include "imageio/SimpleModelLandmarkSink.hpp"
#include "imageio/LandmarkSource.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/SimpleRectLandmarkFormatParser.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/LandmarkMapper.hpp"

#include "logging/LoggerFactory.hpp"


using namespace imageio;
using namespace superviseddescent;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;
using boost::property_tree::ptree;
using boost::filesystem::path;
using boost::lexical_cast;
using render::Mesh;
using cv::Mat;
using cv::Point2f;
using cv::Vec3f;
using cv::Scalar;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;

using render::Mesh;

//from fitter
cv::Mat affineCameraMatrixFromString(std::string cameraMatrixEntries)
{
	cv::Mat cameraMatrix;

	return cameraMatrix;
}
//from fitter
std::string affineCameraMatrixToString(cv::Mat cameraMatrix)
{
	std::string cameraMatrixEntries;
	for (auto i = 0; i < cameraMatrix.rows; ++i) {
		for (auto j = 0; j < cameraMatrix.cols; ++j) {
			cameraMatrixEntries += std::to_string(cameraMatrix.at<float>(i, j));
			cameraMatrixEntries += string(" ");
		}
	}
	return cameraMatrixEntries;
}


template<class T> 
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(cout, " "));
	return os;
}

int main(int argc, char *argv[])
{

    string verboseLevelConsole;
	bool useFileList = false;
	bool useImgs = false;
	bool useDirectory = false;
	vector<path> inputPaths;
	path inputFilelist;
	path inputDirectory;
	vector<path> inputFilenames;
	shared_ptr<ImageSource> imageSource;
	path sdmModelFile;
	path faceDetectorFilename;
	path configFilename;
	path landmarkMappings;
	path outputPath;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"Produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "Specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("config,c", po::value<path>(&configFilename)->required(), 
				"path to a config (.cfg) file")	  
			("input,i", po::value<vector<path>>(&inputPaths)->required(),
				"Input from one or more files, a directory, or a  .lst/.txt-file containing a list of images")
			("model,m", po::value<path>(&sdmModelFile)->required(),
				"An SDM model file to load.")
			("face-detector,f", po::value<path>(&faceDetectorFilename)->required(),
				"Path to an XML CascadeClassifier from OpenCV. Specify either -f or -l.")
            ("landmark-mappings,p", po::value<path>(&landmarkMappings),
				"an optional mapping-file that maps from the input landmarks to landmark identifiers in the model's format")	
			("output,o", po::value<path>(&outputPath)->required(),
				"Output directory for the result images and landmarks.")
		;

		po::positional_options_description p;
		p.add("input", -1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: detect-landmarks [options]" << endl;
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
	Loggers->getLogger("detect-landmarks").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("morphablemodel").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("render").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("fitting").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("fitter").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	
    //Logger appLogger = Loggers->getLogger("detect-landmarks"); //from detection
	Logger appLogger = Loggers->getLogger("fitter"); //from fitting
	
	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));
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


	if (useImgs==true) {
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

	SdmLandmarkModel lmModel = SdmLandmarkModel::load(sdmModelFile);
	SdmLandmarkModelFitting modelFitter(lmModel);


    // Load the face detector:
	cv::CascadeClassifier faceCascade;

    if (!faceCascade.load(faceDetectorFilename.string()))
    {
        appLogger.error("Error loading the face detection model.");
        return EXIT_FAILURE;
    }

    LandmarkMapper landmarkMapper;
	if (!landmarkMappings.empty()) {
		// the user has given a landmark mappings file on the console
		landmarkMapper = LandmarkMapper(landmarkMappings);
	} // Ideas for a better solution: A flag in LandmarkMapper, or polymorphism (IdentityLandmarkMapper), or in Mapper, if mapping empty, return input?, or...?
	
	while (imageSource->next()) {
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + imageSource->getName().string());
		img = imageSource->getImage();
		Mat landmarksImageFitterDetection = img.clone();
		Mat imgGray;
		cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
		vector<cv::Rect> faces;

        float score, notFace = 0.5;
        // face detection
        //faceCascade.detectMultiScale(img, faces, 1.2, 2, 0, cv::Size(50, 50));
        faceCascade.detectMultiScale(img, faces);
        if (faces.empty()) {
            // no face found, output the unmodified image. don't create a file for the (non-existing) landmarks.
            imwrite((outputPath / imageSource->getName().filename()).string(), landmarksImageFitterDetection);
            continue;
        }

		
		// draw the best face candidate (or the face from the face box landmarks)
		cv::rectangle(landmarksImageFitterDetection, faces[0], cv::Scalar(0.0f, 0.0f, 255.0f));

		// fit the model
		Mat modelShape = lmModel.getMeanShape();
		modelShape = modelFitter.alignRigid(modelShape, faces[0]);
		//superviseddescent::drawLandmarks(landmarksImageFitterDetection, modelShape);
		modelShape = modelFitter.optimize(modelShape, imgGray);
		//superviseddescent::drawLandmarks(landmarksImageFitterDetection, modelShape);

		// draw the final result
		superviseddescent::drawLandmarks(landmarksImageFitterDetection, modelShape, Scalar(0.0f, 255.0f, 0.0f));

        // save the image****************************************************************************************************
        //path outputFilename = outputPath / imageSource->getName().filename();
        //imwrite(outputFilename.string(), landmarksImageFitterDetection);
		
		LandmarkCollection lms = lmModel.getAsLandmarks(modelShape);
		LandmarkCollection didLms;
		if (!landmarkMappings.empty()) {
			didLms = landmarkMapper.convert(lms);
		}
		else {
			didLms = lms;
		}
	
		landmarks.clear();
		Mat landmarksImageFitter = img.clone(); // blue rect = the used landmarks
		for (const auto& lm : didLms.getLandmarks()) {
			lm->draw(landmarksImageFitter);
			landmarks.emplace_back(imageio::ModelLandmark(lm->getName(), lm->getPosition2D()));
			cv::rectangle(landmarksImageFitter, cv::Point(cvRound(lm->getX() - 2.0f), cvRound(lm->getY() - 2.0f)), cv::Point(cvRound(lm->getX() + 2.0f), cvRound(lm->getY() + 2.0f)), cv::Scalar(255, 0, 0));
			//cv::putText(landmarksImageFitter, lm->getName(), cv::Point(lm->getX(), lm->getY()), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0.0, 0.0, 255.0));
		}

		// Start affine camera estimation (Aldrian paper)
		Mat affineCamLandmarksProjectionImage = landmarksImageFitter.clone(); // the affine LMs are currently not used (don't know how to render without z-vals)

		// Convert the landmarks to clip-space, and only convert the ones that exist in the model
		vector<imageio::ModelLandmark> landmarksClipSpace;
		for (const auto& lm : landmarks) {
			if (morphableModel.getShapeModel().landmarkExists(lm.getName())) {
				cv::Vec2f clipCoords = render::utils::screenToClipSpace(lm.getPosition2D(), img.cols, img.rows);
				landmarksClipSpace.push_back(imageio::ModelLandmark(lm.getName(), Vec3f(clipCoords[0], clipCoords[1], 0.0f), lm.isVisible()));
			}
		}
		
		Mat affineCam = fitting::estimateAffineCamera(landmarksClipSpace, morphableModel);

		// Render the mean-face landmarks projected using the estimated camera:
		// Todo/Note: Here we render all landmarks. Shouldn't we only render the ones that exist in the model? (see above, landmarksClipSpace)
		for (const auto& lm : landmarks) {
			Vec3f modelPoint;
			try {
				modelPoint = morphableModel.getShapeModel().getMeanAtPoint(lm.getName());
			}
			catch (std::out_of_range& e) {
				continue;
			}
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
		Mat renderedModel = framebuffer.first.clone(); // we save that later, and the framebuffer gets overwritten

		// Extract the texture
		// Todo: check for if hasTexture, we can't do it if the model doesn't have texture coordinates
		Mat textureMap = render::utils::extractTexture(mesh, fullAffineCam, img.cols, img.rows, img, framebuffer.second);

		// Save the extracted texture map (isomap):
        path isomapFilename = outputPath / imageSource->getName().stem();
		isomapFilename += "_isomap.png";
		cv::imwrite(isomapFilename.string(), textureMap);

		// Render the shape-model with the extracted texture from a frontal viewpoint:
		float aspect = static_cast<float>(img.cols) / static_cast<float>(img.rows);
		Mat frontalCam = render::utils::MatrixUtils::createOrthogonalProjectionMatrix(-1.0f * aspect, 1.0f * aspect, -1.0f, 1.0f, 0.1f, 100.0f) * render::utils::MatrixUtils::createScalingMatrix(1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f);
		softwareRenderer.enableTexturing(true);
		auto texture = make_shared<render::Texture>();
		texture->createFromFile(isomapFilename.string());
		softwareRenderer.setCurrentTexture(texture);
		auto frFrontal = softwareRenderer.render(mesh, frontalCam);

		// Write the fitting output files containing:
		// - Camera parameters, fitting parameters, shape coefficients
		ptree fittingFile;
		fittingFile.put("camera", string("affine"));
		fittingFile.put("camera.matrix", affineCameraMatrixToString(fullAffineCam));

		fittingFile.put("imageWidth", img.cols);
		fittingFile.put("imageHeight", img.rows);

		fittingFile.put("fittingParameters.lambda", lambda);

		fittingFile.put("textureMap", isomapFilename.filename().string());
		fittingFile.put("model", config.get_child("morphableModel").get<string>("filename")); // This can throw, but the filename should really exist.

		// alphas:
		fittingFile.put("shapeCoefficients", "");
		for (size_t i = 0; i < fittedCoeffs.size(); ++i) {
			fittingFile.put("shapeCoefficients." + std::to_string(i), fittedCoeffs[i]);
		}

		// Save the fitting file
        path fittingFileName = outputPath / imageSource->getName().stem();
		fittingFileName += ".txt";
		boost::property_tree::write_info(fittingFileName.string(), fittingFile);

		// Additional optional output, as set in the config file:
		if (config.get_child("output", ptree()).get<bool>("copyInputImage", false)) {
            path outInputImage = outputPath / imageSource->getName().filename();
			cv::imwrite(outInputImage.string(), img);
		}
		if (config.get_child("output", ptree()).get<bool>("landmarksImage", false)) {
            path outLandmarksImage = outputPath / imageSource->getName().stem();
			outLandmarksImage += "_landmarks.png";
			cv::imwrite(outLandmarksImage.string(), affineCamLandmarksProjectionImage);
		}
		if (config.get_child("output", ptree()).get<bool>("writeObj", false)) {
            path outMesh = outputPath / imageSource->getName().stem();
			outMesh.replace_extension("obj");
			Mesh::writeObj(mesh, outMesh.string());
		}
		if (config.get_child("output", ptree()).get<bool>("renderResult", false)) {
            path outRenderResult = outputPath / imageSource->getName().stem();
			outRenderResult += "_render.png";
			cv::imwrite(outRenderResult.string(), renderedModel);
		}
		if (config.get_child("output", ptree()).get<bool>("frontalRendering", false)) {
            path outFrontalRenderResult = outputPath / imageSource->getName().stem();
			outFrontalRenderResult += "_render_frontal.png";
			cv::imwrite(outFrontalRenderResult.string(), frFrontal.first);
		}

		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds)+"ms.");
	
		
	}
	
	return 0;
}
