/*
 * ffpDetectAppExp.cpp
 *
 *  Created on: 22.03.2013
 *      Author: Patrik Huber
 */

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
#define _CRTDBG_MAP_ALLOC
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

#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <numeric>

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
#include "boost/lexical_cast.hpp"
#include "boost/interprocess/managed_shared_memory.hpp"
#include "boost/interprocess/allocators/allocator.hpp"
#include "boost/interprocess/containers/vector.hpp"

#include "imageio/ImageSource.hpp"
#include "imageio/FileImageSource.hpp"
#include "imageio/FileListImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/RectLandmark.hpp"
#include "imageio/ModelLandmark.hpp"
#include "imageio/IbugLandmarkFormatParser.hpp"
#include "imageio/LstLandmarkFormatParser.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/NamedLabeledImageSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/FddbLandmarkSink.hpp"

#include "shapemodels/MorphableModel.hpp"
#include "shapemodels/CameraEstimation.hpp"

#include "render/Camera.hpp"
#include "render/SoftwareRenderer.hpp"

#include "logging/LoggerFactory.hpp"
#include "imagelogging/ImageLoggerFactory.hpp"
#include "imagelogging/ImageFileWriter.hpp"

namespace po = boost::program_options;
using namespace std;
using namespace imageio;
using logging::Logger;
using logging::LoggerFactory;
using logging::loglevel;
using imagelogging::ImageLogger;
using imagelogging::ImageLoggerFactory;
using boost::property_tree::ptree;
using boost::property_tree::info_parser::read_info;
using boost::filesystem::path;
using boost::lexical_cast;


void drawFfpsCircle(Mat image, pair<string, Point2f> landmarks)
{
	cv::Point center(cvRound(landmarks.second.x), cvRound(landmarks.second.y));
	int radius = cvRound(3);
	circle(image, center, 1, cv::Scalar(0,255,0), 1, 8, 0 );	// draw the circle center
	circle(image, center, radius, cv::Scalar(0,0,255), 1, 8, 0 );	// draw the circle outline

}

void drawFfpsText(Mat image, pair<string, Point2f> landmarks)
{
	cv::Point center(cvRound(landmarks.second.x), cvRound(landmarks.second.y));
	std::ostringstream text;
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 0.7;
	int thickness = 1;  
	text << landmarks.first << std::ends;
	putText(image, text.str(), center, fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
	text.str("");
}

void doNothing() {};

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
	return os;
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	string verboseLevelImages;
	bool useFileList = false;
	bool useImgs = false;
	bool useDirectory = false;
	bool useGroundtruth = false;
	vector<path> inputPaths;
	path inputFilelist;
	path inputDirectory;
	vector<path> inputFilenames;
	path configFilename;
	shared_ptr<ImageSource> imageSource;
	path outputPicsDir; // TODO: ImageLogger vs ImageSinks? (see AdaptiveTracking.cpp)
	path groundtruthDir; // TODO: Make more dynamic wrt landmark format. a) What about the loading-flags (1_Per_Folder etc) we have? b) Expose those flags to cmdline? c) Make a LmSourceLoader and he knows about a LM_TYPE (each corresponds to a Parser/Loader class?)
	// TODO Also, sometimes we might have the face-box annotated but not LMs, sometimes only LMs and no Facebox.
	string groundtruthType;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("verbose-images,w", po::value<string>(&verboseLevelImages)->implicit_value("INTERMEDIATE")->default_value("FINAL","write images with FINAL loglevel or below."),
				  "specify the verbosity of the image output: FINAL, INTERMEDIATE, INFO, DEBUG or TRACE")
			("config,c", po::value<path>(&configFilename)->required(), 
				"path to a config (.cfg) file")
			("input,i", po::value<vector<path>>(&inputPaths)->required(), 
				"input from one or more files, a directory, or a  .lst/.txt-file containing a list of images")
			("groundtruth,g", po::value<path>(&groundtruthDir), 
				"load ground truth landmarks from the given folder along with the images and output statistics of the detection results")
			("groundtruth-type,t", po::value<string>(&groundtruthType), 
				"specify the type of landmarks to load: lst, ibug")
			("output-dir,o", po::value<path>(&outputPicsDir)->default_value("."),
				"output directory for the result images")
		;

		po::positional_options_description p;
		p.add("input", -1);
		
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << "Usage: ffpDetectApp [options]\n";
			cout << desc;
			return EXIT_SUCCESS;
		}
	
		if (vm.count("groundtruth")) {
			useGroundtruth = true;
			if (!vm.count("groundtruth-type")) {
				cout << "You have specified to use ground truth. Please also specify the type of the landmarks to load via --groundtruth-type or -t." << endl;
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
	imagelogging::loglevel imageLogLevel;
	if(boost::iequals(verboseLevelImages, "FINAL")) imageLogLevel = imagelogging::loglevel::FINAL;
	else if(boost::iequals(verboseLevelImages, "INTERMEDIATE")) imageLogLevel = imagelogging::loglevel::INTERMEDIATE;
	else if(boost::iequals(verboseLevelImages, "INFO")) imageLogLevel = imagelogging::loglevel::INFO;
	else if(boost::iequals(verboseLevelImages, "DEBUG")) imageLogLevel = imagelogging::loglevel::DEBUG;
	else if(boost::iequals(verboseLevelImages, "TRACE")) imageLogLevel = imagelogging::loglevel::TRACE;
	else {
		cout << "Error: Invalid image loglevel." << endl;
		return EXIT_FAILURE;
	}
	
	Loggers->getLogger("classification").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("imageprocessing").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("detection").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("shapemodels").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("ffpDetectApp").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("ffpDetectApp");

	appLogger.debug("Verbose level for console output: " + logging::loglevelToString(logLevel));
	appLogger.debug("Verbose level for image output: " + imagelogging::loglevelToString(imageLogLevel));
	appLogger.debug("Using config: " + configFilename.string());
	appLogger.debug("Using output directory: " + outputPicsDir.string());
	if(outputPicsDir == ".") {
		appLogger.info("Writing output images into current directory.");
	}

	ImageLoggers->getLogger("detection").addAppender(make_shared<imagelogging::ImageFileWriter>(imageLogLevel, outputPicsDir));
	ImageLoggers->getLogger("app").addAppender(make_shared<imagelogging::ImageFileWriter>(imageLogLevel, outputPicsDir / "final"));

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
	if (useGroundtruth) {
		vector<path> groundtruthDirs; groundtruthDirs.push_back(groundtruthDir); // Todo: Make cmdline use a vector<path>
		shared_ptr<LandmarkFormatParser> landmarkFormatParser;
		if(boost::iequals(groundtruthType, "lst")) {
			landmarkFormatParser = make_shared<LstLandmarkFormatParser>();
			landmarkSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(imageSource, string(), GatherMethod::SEPARATE_FILES, groundtruthDirs), landmarkFormatParser);
		} else if(boost::iequals(groundtruthType, "ibug")) {
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
		read_info(configFilename.string(), pt);
	} catch(const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	string morphableModelFile;
	string morphableModelVertexMappingFile;
	shapemodels::MorphableModel mm;
	
	try {
		ptree ptFeaturePointValidation = pt.get_child("featurePointValidation");
		morphableModelFile = ptFeaturePointValidation.get<string>("morphableModel");
		morphableModelVertexMappingFile = ptFeaturePointValidation.get<string>("morphableModelVertexMapping");
		mm = shapemodels::MorphableModel::loadScmModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\SurreyLowResGuosheng\\NON3448\\ShpVtxModelBin_NON3448.scm", "C:\\Users\\Patrik\\Documents\\GitHub\\featurePoints_SurreyScm.txt");
	} catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}
	
	FddbLandmarkSink landmarkSink("annotatedList.txt");
	landmarkSink.open(outputPicsDir.string() + "/final/" + "detectedFaces.txt");
	

	// ---

	boost::interprocess::managed_shared_memory managed_shm(boost::interprocess::open_only, "CameraInputMem");
	//catch (boost::interprocess::interprocess_exception& e)

	typedef boost::interprocess::allocator<uchar, boost::interprocess::managed_shared_memory::segment_manager> CharAllocator;
	typedef boost::interprocess::vector<uchar, CharAllocator> IpcVec;
	const CharAllocator alloc_inst(managed_shm.get_segment_manager());

	typedef boost::interprocess::allocator<float, boost::interprocess::managed_shared_memory::segment_manager> FloatAllocator;
	typedef boost::interprocess::vector<float, FloatAllocator> FloatVec;
	vector<float> landmarksX;
	vector<float> landmarksY;

	shapemodels::CameraEstimation cameraEstimation(mm);
	vector<imageio::ModelLandmark> landmarks;

	cv::namedWindow("win");
	while (true) {
		std::pair<IpcVec*, std::size_t> s = managed_shm.find<IpcVec>("EncodedImage"); // check for s.second != 1?
		vector<uchar> vimg;
		for (const auto& e : *s.first) {
			vimg.push_back(e);
		}	
		cv::Mat img2 = cv::imdecode(vimg, -1); // opt.: 3rd: a *Mat for no reallocation every frame
		if (img2.rows <= 0 || img2.cols <= 0) {
			continue;
		}
		std::pair<FloatVec*, std::size_t> sX = managed_shm.find<FloatVec>("LandmarksX"); // check for s.second != 1?
		std::pair<FloatVec*, std::size_t> sY = managed_shm.find<FloatVec>("LandmarksY"); // check for s.second != 1?
		landmarksX.clear();
		landmarksY.clear();
		if (sX.second == 1 && sY.second == 1) {
			for (const auto& lm : *sX.first) {
				landmarksX.push_back(lm);
			}
			for (const auto& lm : *sY.first) {
				landmarksY.push_back(lm);
			}
		
			for (int i = 0; i < landmarksX.size(); ++i) {
				cv::circle(img2, cv::Point((int)landmarksX[i], (int)landmarksY[i]), 1, cv::Scalar(0,255,0), -1);
			}
			landmarks.clear();
			landmarks.emplace_back(imageio::ModelLandmark("right.eye.corner_outer", landmarksX[19], landmarksY[19]));
			landmarks.emplace_back(imageio::ModelLandmark("right.eye.corner_inner", landmarksX[22], landmarksY[22]));
			landmarks.emplace_back(imageio::ModelLandmark("left.eye.corner_outer", landmarksX[28], landmarksY[28]));
			landmarks.emplace_back(imageio::ModelLandmark("left.eye.corner_inner", landmarksX[25], landmarksY[25]));
			landmarks.emplace_back(imageio::ModelLandmark("center.nose.tip", landmarksX[13], landmarksY[13]));
			landmarks.emplace_back(imageio::ModelLandmark("right.lips.corner", landmarksX[31], landmarksY[31]));
			landmarks.emplace_back(imageio::ModelLandmark("left.lips.corner", landmarksX[37], landmarksY[37]));
			
			int max_d = std::max(img2.rows, img2.cols); // should be the focal length? (don't forget the aspect ratio!). TODO Read in Hartley-Zisserman what this is
			//int max_d = 700;
			Mat camMatrix = (cv::Mat_<double>(3,3) << max_d, 0,		img2.cols/2.0,
				0,	 max_d, img2.rows/2.0,
				0,	 0,		1.0);

			pair<Mat, Mat> rotTransRodr = cameraEstimation.estimate(landmarks, camMatrix);
			Mat rvec = rotTransRodr.first;
			Mat tvec = rotTransRodr.second;

			Mat rotation_matrix(3, 3, CV_64FC1);
			cv::Rodrigues(rvec, rotation_matrix);
			rotation_matrix.convertTo(rotation_matrix, CV_32FC1);
			Mat translation_vector = tvec;
			translation_vector.convertTo(translation_vector, CV_32FC1);

			camMatrix.convertTo(camMatrix, CV_32FC1);

			for (const auto& p : landmarks) {
				cv::rectangle(img2, cv::Point(cvRound(p.getX()-2.0f), cvRound(p.getY()-2.0f)), cv::Point(cvRound(p.getX()+2.0f), cvRound(p.getY()+2.0f)), cv::Scalar(255, 0, 0));
			}
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
				points3d.emplace_back(mm.getShapeModel().getMeanAtPoint(landmark.getName()));
			}
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
				Point2f v4p2d(v4p.x/v4p.z, v4p.y/v4p.z); // if != 0
				Point3f v4p_homo(v4_mat(cv::Range(0, 3), cv::Range(0, 1)));
				Point2f v4p2d_homo(v4p_homo.x/v4p_homo.z, v4p_homo.y/v4p_homo.z); // if != 0
				cv::rectangle(img2, cv::Point(cvRound(v4p2d_homo.x-2.0f), cvRound(v4p2d_homo.y-2.0f)), cv::Point(cvRound(v4p2d_homo.x+2.0f), cvRound(v4p2d_homo.y+2.0f)), cv::Scalar(255, 0, 0));
			}

			std::shared_ptr<render::Mesh> meshToDraw = std::make_shared<render::Mesh>(mm.getMean());

			const float aspect = (float)img2.cols/(float)img2.rows; // 640/480
			render::Camera camera(Vec3f(0.0f, 0.0f, 0.0f), /*horizontalAngle*/0.0f*(CV_PI/180.0f), /*verticalAngle*/0.0f*(CV_PI/180.0f), render::Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, /*zNear*/-0.1f, /*zFar*/-100.0f));
			render::SoftwareRenderer r(img2.cols, img2.rows, camera); // 640, 480
			//r.setModelTransform(render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f));
			r.setObjectToScreenTransform(intrinsicCameraMatrix * extrinsicCameraMatrix);
			r.draw(meshToDraw, nullptr);
			Mat buff = r.getImage();
			Mat buffWithoutAlpha;
			//buff.convertTo(buffWithoutAlpha, CV_BGRA2BGR);
			cvtColor(buff, buffWithoutAlpha, cv::COLOR_BGRA2BGR);
			Mat weighted = img2.clone(); // get the right size
			cv::addWeighted(img2, 0.7, buffWithoutAlpha, 0.3, 0.0, weighted);
			//return std::make_pair(translation_vector, rotation_matrix);
			img2 = weighted;
		}
		cv::imshow("win", img2);
		cv::waitKey(5);
	}

	// ---


	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	while(labeledImageSource->next()) {
		start = std::chrono::system_clock::now();
		appLogger.info("Starting to process " + labeledImageSource->getName().string());
		img = labeledImageSource->getImage();
	
		end = std::chrono::system_clock::now();
		int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		appLogger.debug("Finished face-detection. Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");
	
		// Log the image with the max positive of every feature
		ImageLogger appImageLogger = ImageLoggers->getLogger("app");
		appImageLogger.setCurrentImageName(labeledImageSource->getName().stem().string());
		appImageLogger.intermediate(img, doNothing, "AllFfpMaxPos");

		LandmarkCollection groundtruth = labeledImageSource->getLandmarks();

		end = std::chrono::system_clock::now();
		elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		appLogger.info("Finished processing " + labeledImageSource->getName().string() + ". Elapsed time: " + lexical_cast<string>(elapsed_mseconds) + "ms.\n");
		
		//landmarkSink.add(labeledImageSource->getName().string(), faces, scores);

	}
	landmarkSink.close();

	appLogger.info("Finished!");

	return 0;
}
