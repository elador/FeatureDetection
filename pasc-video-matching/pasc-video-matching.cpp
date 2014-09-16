/*
 * pasc-video-matching.cpp
 *
 *  Created on: 10.09.2014
 *      Author: Patrik Huber
 *
 * Example:
 * pasc-video-matching -c ../../FeatureDetection/pasc-video-matching/share/configs/default.cfg -s ../../data/PaSC/sigset.xml -d ../../data/PaSC/video
 *   
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <thread>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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

#include "fitting/AffineCameraEstimation.hpp"
#include "fitting/OpenCVCameraEstimation.hpp"
#include "fitting/LinearShapeFitting.hpp"

#include "render/SoftwareRenderer.hpp"
#include "render/MeshUtils.hpp"
#include "render/utils.hpp"

#include "facerecognition/utils.hpp"

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
#include "imageio/LandmarkMapper.hpp"

#include "logging/LoggerFactory.hpp"

#include "ThreadPool.hpp"

// For frame-extract:
#include <iomanip>
#include "imageio/SimpleModelLandmarkSink.hpp"
#include "imageio/PascVideoEyesLandmarkFormatParser.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/optional.hpp"

using namespace imageio;
namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using facerecognition::FaceRecord;
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

/**
 * Extracts...
 * Todo/Note: I think it would be much better to pass around objects (i.e. cv::Mats) instead of paths.
 *
 * @param[in] in Todo
 * @return Todo. Paths or only filenames? Because we give the output path already.
 * (Todo proper doxygen) Throws a std::runtime_error when...
 */
vector<path> extractFrames(path videoFilename, path outputPath, boost::optional<shared_ptr<imageio::NamedLandmarkSource>> landmarkSource)
{
	boost::filesystem::create_directory(outputPath);
	vector<path> frames;

	// Output landmarks for the extracted frames:
	imageio::SimpleModelLandmarkSink landmarkSink;

	cv::VideoCapture cap(videoFilename.string());
	if (!cap.isOpened())
		throw std::runtime_error("Err opening cam");

	int frameCount = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_COUNT));
	if (frameCount == 0) {
		// error, property not supported.
		throw std::runtime_error("cv::VideoCapture::get(CV_CAP_PROP_FRAME_COUNT) not supported.");
	}

	int frameToExtract = 0;

	// Get and write the frame image:
	Mat img;
	cap.set(CV_CAP_PROP_POS_FRAMES, frameToExtract); // starts at 0
	cap >> img;
	path fn = outputPath / videoFilename.filename();
	frameToExtract++; // PaSC starts naming them from 1
	std::ostringstream frameNumPadded;
	frameNumPadded << std::setw(3) << std::setfill('0') << frameToExtract;
	fn.replace_extension(frameNumPadded.str() + ".png");
	cv::imwrite(fn.string(), img);

	frames.emplace_back(fn); // todo

	// Get and write the frames landmarks:
	string pascFrameName = videoFilename.stem().string() + "/" + videoFilename.stem().string() + "-" + frameNumPadded.str() + ".jpg";
	imageio::LandmarkCollection landmarks;
	// we only want the eyes, not the face (not a ModelLandmark):
	landmarks.insert(landmarkSource.get()->get(pascFrameName).getLandmark("le")); // the first get() is from the boost::optional
	landmarks.insert(landmarkSource.get()->get(pascFrameName).getLandmark("re"));
	fn.replace_extension(".txt");
	landmarkSink.add(landmarks, fn);

	return frames;
}

int main(int argc, char *argv[])
{
	string verboseLevelConsole;
	path querySigsetFilename;
	path targetSigsetFilename;
	path dataDirectory;
	path configFilename;
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
			("query-sigset,q", po::value<path>(&querySigsetFilename)->required(),
				"query sigset file")
			("target-sigset,t", po::value<path>(&targetSigsetFilename)->required(),
				"target sigset file")
			("data,d", po::value<path>(&dataDirectory)->required(),
				"database videos")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: pasc-video-matching [options]" << endl;
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
	Loggers->getLogger("facerecognition").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("pasc-video-matching").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("pasc-video-matching");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));
	appLogger.debug("Using config: " + configFilename.string());

	ThreadPool threadPool(4); // Todo: Cmd-line parameter

	// Read the query and target sigsets
	// For every video (not pair, because a video can be used more than once), spawn a thread (thread pool? max?)
	//	Preprocess the video, e.g. extract frames (optional)
	// We're now left with one or more frames or the whole video
	// For every video:
	//	LMDet/Fitting now: Input: This. Output: One pose-normalised frontal image.
	//	// Maybe combine both to 1 loop. After all they might be coupled?
	// For every pair:
	//	Launch a thread, as above
	//	Match the two pose-normalised images (Note: Allow the possibility to only pose-normalise one image)
	//	Output: Score.
	// Write all scores to the MySQL DB or a BEE matrix

	auto querySigset = facerecognition::utils::readPascSigset(querySigsetFilename);
	auto targetSigset = facerecognition::utils::readPascSigset(targetSigsetFilename);
	
	// We find the unique elements (video or image files) of both sigsets and (pre)process them:
	// Note: Really? There might be some cases where we want to preprocess the query different from the gallery?
/*	auto allDataItems = querySigset;
	allDataItems.insert(end(allDataItems), begin(targetSigset), end(targetSigset));
	auto it = std::unique(begin(allDataItems), end(allDataItems), [](FaceRecord lhs, FaceRecord rhs) { return lhs.identifier == rhs.identifier });
	allDataItems.resize(std::distance(begin(allDataItems), it));
	*/

	// Create the output, query and target directories. If they already exist, nothing happens.
	boost::filesystem::create_directory(outputPath);
	boost::filesystem::create_directory(outputPath / "query");
	boost::filesystem::create_directory(outputPath / "target");

	// TODO: I think this should be separate - the function should only return the frames, not the LMs.
	// Keep functions simple. But then we have to find out the frame-number using parsing... Maybe return a pair<frameNr, path>?
	// But a frame-extraction algorithm may encompass detecting/using landmarks, so...
	// Read the PaSC video landmarks:
	shared_ptr<imageio::NamedLandmarkSource> landmarkSource;
	vector<path> groundtruthDirs{ path(R"(C:\Users\Patrik\Documents\GitHub\data\PaSC\pasc_video_pittpatt_detections_p.csv)") };
	shared_ptr<imageio::LandmarkFormatParser> landmarkFormatParser;
	string groundtruthType = "PaSC-video-PittPatt-detections";
	if (boost::iequals(groundtruthType, "PaSC-video-PittPatt-detections")) { // Todo/Note: Not sure this is working?
		landmarkFormatParser = make_shared<imageio::PascVideoEyesLandmarkFormatParser>();
		landmarkSource = make_shared<imageio::DefaultNamedLandmarkSource>(imageio::LandmarkFileGatherer::gather(nullptr, ".csv", imageio::GatherMethod::SEPARATE_FILES, groundtruthDirs), landmarkFormatParser);
	}
	else {
		throw std::runtime_error("Invalid ground-truth landmarks type.");
	}

	vector<vector<path>> queryFrames;
	// We could/have to do it for the target frames as well, depending on the protocol
	{
		vector<std::future<vector<path>>> frameFutures;
		for (const auto& q : querySigset) {
			frameFutures.emplace_back(threadPool.enqueue(extractFrames, dataDirectory / q.dataPath, outputPath / "query", landmarkSource));
		}
		for (auto& f : frameFutures) {
			queryFrames.emplace_back(f.get());
		}
	}

	// Fit every query and target image:
	// Read the 3DMM config file
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
	
	float lambda = config.get_child("fitting", ptree()).get<float>("lambda", 15.0f);

	for (const auto& q : queryFrames) {

	}

	return EXIT_SUCCESS;
}
