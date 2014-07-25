/*
* evaluate-landmarks.cpp
*
*  Created on: 03.02.2014
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
#include <numeric>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/algorithm/string.hpp"

#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/EmptyLandmarkSource.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/IbugLandmarkFormatParser.hpp"
#include "imageio/SimpleModelLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
namespace po = boost::program_options;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using boost::property_tree::ptree;
using boost::filesystem::path;
using cv::Mat;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;

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
	path configFilename, inputPath, groundtruthPath, outputFilename;
	string inputType, groundtruthType;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO", "show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("config,c", po::value<path>(&configFilename)->required(),
				"path to a config (.cfg) file")
			("input,i", po::value<path>(&inputPath)->required(),
				"input landmarks")
			("input-type,s", po::value<string>(&inputType)->required(),
				"specify the type of landmarks to load: simple")
			("groundtruth,g", po::value<path>(&groundtruthPath)->required(),
				"groundtruth landmarks")
			("groundtruth-type,t", po::value<string>(&groundtruthType)->required(),
				"specify the type of landmarks to load: ibug")
			("output,o", po::value<path>(&outputFilename),
				"output filename to write the normalized landmark errors to")
			;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: evaluate-landmarks [options]\n";
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
	if (boost::iequals(verboseLevelConsole, "PANIC")) logLevel = LogLevel::Panic;
	else if (boost::iequals(verboseLevelConsole, "ERROR")) logLevel = LogLevel::Error;
	else if (boost::iequals(verboseLevelConsole, "WARN")) logLevel = LogLevel::Warn;
	else if (boost::iequals(verboseLevelConsole, "INFO")) logLevel = LogLevel::Info;
	else if (boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = LogLevel::Debug;
	else if (boost::iequals(verboseLevelConsole, "TRACE")) logLevel = LogLevel::Trace;
	else {
		cout << "Error: Invalid loglevel." << endl;
		return EXIT_FAILURE;
	}

	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("evaluate-landmarks").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("evaluate-landmarks");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));
	appLogger.debug("Using config: " + configFilename.string());

	// output: the avg-err plus a file with the differences (+comments?) for matlab.

	// Load the ground truth
	shared_ptr<NamedLandmarkSource> groundtruthSource;
	vector<path> groundtruthDirs{ groundtruthPath };
	shared_ptr<LandmarkFormatParser> landmarkFormatParser;
	if (boost::iequals(groundtruthType, "ibug")) {
		landmarkFormatParser = make_shared<IbugLandmarkFormatParser>();
		groundtruthSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(nullptr, ".pts", GatherMethod::SEPARATE_FOLDERS, groundtruthDirs), landmarkFormatParser);
	}
	else {
		appLogger.error("Error: Invalid ground-truth landmarks type.");
		return EXIT_FAILURE;
	}

	// Load the landmarks to compare against, usually automatically detected landmarks:
	shared_ptr<NamedLandmarkSource> inputSource;
	vector<path> inputDirs{ inputPath };
	shared_ptr<LandmarkFormatParser> landmarkFormatParserInput;
	if (boost::iequals(inputType, "ibug")) {
		landmarkFormatParserInput = make_shared<IbugLandmarkFormatParser>();
		inputSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(nullptr, ".pts", GatherMethod::SEPARATE_FOLDERS, inputDirs), landmarkFormatParserInput);
	}
	else if (boost::iequals(inputType, "simple")) {
		landmarkFormatParserInput = make_shared<SimpleModelLandmarkFormatParser>();
		inputSource = make_shared<DefaultNamedLandmarkSource>(LandmarkFileGatherer::gather(nullptr, ".txt", GatherMethod::SEPARATE_FOLDERS, inputDirs), landmarkFormatParserInput);
	}
	else {
		appLogger.error("Error: Invalid input landmarks type.");
		return EXIT_FAILURE;
	}

	ptree pt;
	try {
		boost::property_tree::info_parser::read_info(configFilename.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error(error.what());
		return EXIT_FAILURE;
	}

	string rightEye;
	string leftEye;
	try {
		ptree ptParameters = pt.get_child("interEyeDistance");
		rightEye = ptParameters.get<string>("rightEye");
		leftEye = ptParameters.get<string>("leftEye");
	}
	catch (const boost::property_tree::ptree_error& error) {
		appLogger.error("Parsing config: " + string(error.what()));
		return EXIT_FAILURE;
	}

	// Process the interEyeDistance landmarks - one or two identifiers might be given
	vector<string> rightEyeIdentifiers;
	boost::split(rightEyeIdentifiers, rightEye, boost::is_any_of(" "));
	vector<string> leftEyeIdentifiers;
	boost::split(leftEyeIdentifiers, leftEye, boost::is_any_of(" "));

	// Create the output file if one was specified on the command line
	std::ofstream outputFile;
	if (!outputFilename.empty()) {
		outputFile.open(outputFilename.string());
		if (!outputFile.is_open()) {
			appLogger.error("An output filename was specified, but the file could not be created: " + outputFilename.string());
			return EXIT_FAILURE;
		}
	}

	vector<float> differences;

	appLogger.info("Starting to process...");
	int totalImages = 0;

	// Loop over the input landmarks
	while (inputSource->next())	{
		path filename = inputSource->getName();
		LandmarkCollection detected = inputSource->getLandmarks();
		LandmarkCollection groundtruth;
		try {
			groundtruth = groundtruthSource->get(filename);
		}
		catch (std::out_of_range& e) {
			appLogger.warn("No groundtruth for this input file available. Skipping this file. This is probably not expected.");
			continue;
		}

		// map both landmarks to a common format:
		// not necessary at the moment, both have ibug IDs

		// Calculate the inter-eye distance of the groundtruth face. Which landmarks to take for that is specified in the config, it
		// might be one or two, and we calculate the average if them (per eye). For example, it might be the outer eye-corners.
		cv::Vec2f rightEyeCenter(0.0f, 0.0f);
		for (const auto& rightEyeIdentifyer : rightEyeIdentifiers) {
			rightEyeCenter += groundtruth.getLandmark(rightEyeIdentifyer)->getPosition2D();
		}
		rightEyeCenter /= static_cast<float>(rightEyeIdentifiers.size());
		cv::Vec2f leftEyeCenter(0.0f, 0.0f);
		for (const auto& leftEyeIdentifyer : leftEyeIdentifiers) {
			leftEyeCenter += groundtruth.getLandmark(leftEyeIdentifyer)->getPosition2D();
		}
		leftEyeCenter /= static_cast<float>(leftEyeIdentifiers.size());

		cv::Scalar interEyeDistance = cv::norm(rightEyeCenter, leftEyeCenter, cv::NORM_L2);

		// See if we discard the detection result:
		// Ideally, we'd use the detected V&J face box here and skip any face with w or h < 20.
		// If the actual detection and evaluation were in one program, this would also allow us to select the correct box for landmark detection, if V&J finds several. We would then select the one where the box-centers are closest.
		// But as we don't have it (without further modifications), we use the face box approximated from the landmarks.
		cv::Rect groundtruthFacebox = getBoundingBox(groundtruth);
		cv::Rect detectedFacebox = getBoundingBox(detected);
		cv::Vec2f groundtruthCenter(groundtruthFacebox.x + groundtruthFacebox.width / 2.0f, groundtruthFacebox.y + groundtruthFacebox.height / 2.0f);
		cv::Vec2f detectedCenter(detectedFacebox.x + detectedFacebox.width / 2.0f, detectedFacebox.y + detectedFacebox.height / 2.0f);

		cv::Scalar distance = cv::norm(groundtruthCenter, detectedCenter, cv::NORM_L2);

		// Compare groundtruth and detected landmarks
		if (!outputFilename.empty()) {
			// write out the filename
			outputFile << "# " << inputSource->getName() << endl;
		}
		for (const auto& detectedLandmark : detected.getLandmarks()) {

			shared_ptr<Landmark> groundtruthLandmark = groundtruth.getLandmark(detectedLandmark->getName()); // Todo: Handle case when LM not found
			cv::Point2f gt = groundtruthLandmark->getPoint2D();
			cv::Point2f det = detectedLandmark->getPoint2D();

			float dx = (gt.x - det.x);
			float dy = (gt.y - det.y);
			float difference = std::sqrt(dx*dx + dy*dy);
			difference = difference / interEyeDistance[0]; // normalize by the IED

			if (!outputFilename.empty()) {
				outputFile << difference << " # " << detectedLandmark->getName() << endl;
			}
			differences.push_back(difference);
		} // for each landmark
		++totalImages;
	} // for each image
	
	// close the file if we had opened it before
	if (!outputFilename.empty()) {
		outputFile.close();
	}

	appLogger.info("Finished processing all input landmarks.");
	appLogger.info("Total number of images: " + std::to_string(totalImages));

	float averageError = std::accumulate(begin(differences), end(differences), 0.0f) / static_cast<float>(differences.size());
	appLogger.info("Average error (normalized by inter-eye distance) over all landmarks and all images: " + std::to_string(averageError));

	return 0;
}
