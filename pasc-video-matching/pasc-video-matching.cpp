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

int main(int argc, char *argv[])
{
	
	string verboseLevelConsole;
	path sigsetFilename;
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
			("sigset,s", po::value<path>(&sigsetFilename)->required(),
				"video2video sigset file")
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

	// Read the sigset
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

	auto sigset = facerecognition::utils::readSigset(sigsetFilename);

	
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
	
	//float lambda = config.get_child("fitting", ptree()).get<float>("lambda", 15.0f);
	return EXIT_SUCCESS;
}
