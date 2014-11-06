/*
 * fvsdk-enrol-all-videos-one-fir-per-video.cpp
 *
 *  Created on: 05.11.2014
 *      Author: Patrik Huber
 *
 * Example:
 * fvsdk-enrol-all-videos ...
 *
 * Note: For this app, in the FVSDK config tool, the
 * value of FRSDK.ComparisonAlgortihm.AlgorithmVersion.B8.MaxNumOfClusters
 * should be set to around 300. (Default: 5)
 * Otherwise, only FIRs of 5 frames will be stored in one FIR file.
 *   
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <random>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/text_oarchive.hpp"
//#include "boost/archive/binary_iarchive.hpp"
//#include "boost/archive/binary_oarchive.hpp"

#include "boost/serialization/vector.hpp"
#include "boost/serialization/optional.hpp"
#include "boost/serialization/utility.hpp"

#include "imageio/MatSerialization.hpp"
#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"
#include "facerecognition/FaceVacsEngine.hpp"
#include "facerecognition/frsdk.hpp"

#include "facerecognition/ThreadPool.hpp"

#include "frsdk/image.h"
#include "frsdk/sample.h"
#include "frsdk/enroll.h"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using cv::Mat;
using boost::filesystem::path;
using boost::lexical_cast;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::vector;
using std::string;

//#include <boost/config.hpp>
//#include <boost/filesystem/path.hpp>
//#include <boost/serialization/level.hpp>
#include "facerecognition/PathSerialization.hpp"


std::ostream&
operator<<(std::ostream& o, const FRsdk::Position& p)
{
	o << "[" << p.x() << ", " << p.y() << "]";
	return o;
}

namespace {
	class InvalidFIRAccessError : public std::exception
	{
	public:
		InvalidFIRAccessError() throw() :
			msg("Trying to access invalid FIR") {}
		~InvalidFIRAccessError() throw() { }
		const char* what() const throw() { return msg.c_str(); }
	private:
		std::string msg;
	};
}

// the concrete feedback which prints to stdout 
class EnrolCoutFeedback : public FRsdk::Enrollment::FeedbackBody
{
public:
	EnrolCoutFeedback(const std::string& firFilename)
		: firFN(firFilename), firvalid(false) { }
	~EnrolCoutFeedback() {}

	// the feedback interface
	void start() {
		firvalid = false;
		std::cout << "start" << std::endl;
	}

	void processingImage(const FRsdk::Image& img)
	{
		std::cout << "processing image[" << img.name() << "]" << std::endl;
	}

	void eyesFound(const FRsdk::Eyes::Location& eyeLoc)
	{
		std::cout << "found eyes at [" << eyeLoc.first
			<< " " << eyeLoc.second << "; confidences: "
			<< eyeLoc.firstConfidence << " "
			<< eyeLoc.secondConfidence << "]" << std::endl;
	}

	void eyesNotFound()
	{
		std::cout << "eyes not found" << std::endl;
	}

	void sampleQualityTooLow() {
		std::cout << "sampleQualityTooLow" << std::endl;
	}


	void sampleQuality(const float& f) {
		std::cout << "Sample Quality: " << f << std::endl;
	}

	void success(const FRsdk::FIR& fir_)
	{
		fir = new FRsdk::FIR(fir_);
		std::cout
			<< "successful enrollment";
		if (firFN != std::string("")) {

			std::cout << " FIR[filename,id,size] = [\""
				<< firFN.c_str() << "\",\"" << (fir->version()).c_str() << "\","
				<< fir->size() << "]";
			// write the fir
			std::ofstream firOut(firFN.c_str(),
				std::ios::binary | std::ios::out | std::ios::trunc);
			firOut << *fir;
		}
		firvalid = true;
		std::cout << std::endl;
	}

	void failure() { std::cout << "failure" << std::endl; }

	void end() { std::cout << "end" << std::endl; }

	const FRsdk::FIR& getFir() const {
		// call only if success() has been invoked    
		if (!firvalid)
			throw InvalidFIRAccessError();

		return *fir;
	}

	bool firValid() const {
		return firvalid;
	}

private:
	FRsdk::CountedPtr<FRsdk::FIR> fir;
	std::string firFN;
	bool firvalid;
};

void enrolImageSet(path videoName, std::shared_ptr<facerecognition::FaceVacsEngine> faceRecEngine, path firPath)
{

	auto frames = facerecognition::utils::getFrames(videoName);

	FRsdk::SampleSet enrollmentImages;
	//for (size_t frameNum = 0; frameNum < 2; ++frameNum)
	for (size_t frameNum = 0; frameNum < frames.size(); ++frameNum)
	{
		Loggers->getLogger("app").debug("Processing frame " + std::to_string(frameNum + 1));

		auto image = facerecognition::matToFRsdkImage(frames[frameNum]);

		FRsdk::Face::Finder faceFinder(*faceRecEngine->getConfigurationInstance().get());
		FRsdk::Eyes::Finder eyesFinder(*faceRecEngine->getConfigurationInstance().get());
		float mindist = 0.005f; // minEyeDist, def = 0.1; me: 0.01; philipp: 0.005
		float maxdist = 0.3f; // maxEyeDist, def = 0.4; me: 0.3; philipp: 0.4
		FRsdk::Face::LocationSet faceLocations = faceFinder.find(image, mindist, maxdist);
		if (faceLocations.empty()) {
			continue;
		}
		// We just use the first found face to detect eyes:
		FRsdk::Eyes::LocationSet eyesLocations = eyesFinder.find(image, *faceLocations.begin());
		if (eyesLocations.empty()) {
			continue;
		}
		auto foundEyes = eyesLocations.begin(); // We just use the first found eyes
		auto firstEye = FRsdk::Position(foundEyes->first.x(), foundEyes->first.y()); // first eye (image left-most), second eye (image right-most)
		auto secondEye = FRsdk::Position(foundEyes->second.x(), foundEyes->second.y());
		auto sample = FRsdk::Sample(image);
		sample.annotate(FRsdk::Eyes::Location(firstEye, secondEye, foundEyes->firstConfidence, foundEyes->secondConfidence));

		enrollmentImages.push_back(sample);
	}

	// create an enrollment processor
	FRsdk::Enrollment::Processor proc(*faceRecEngine->getConfigurationInstance().get());
	// create the needed interaction instances
	FRsdk::Enrollment::Feedback feedback(new EnrolCoutFeedback(firPath.string()));
	// do the enrollment
	proc.process(begin(enrollmentImages), end(enrollmentImages), feedback);
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path inputDirectoryVideos;
	path sigsetFile;
	path landmarksMetadata;
	path outputPath;
	path fvsdkConfig;
	int numThreads;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("sigset,s", po::value<path>(&sigsetFile)->required(),
				  "PaSC video training query sigset")
			("data,d", po::value<path>(&inputDirectoryVideos)->required(),
				"path to the training videos")
			("metadata,m", po::value<path>(&landmarksMetadata)->required(),
				"landmarks for the training videos in boost::serialization text format")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
			("threads,t", po::value<int>(&numThreads)->default_value(4),
				"path to an output folder")
			("fvsdk-config,c", po::value<path>(&fvsdkConfig)->default_value(R"(C:\FVSDK_8_9_5\etc\frsdk.cfg)"),
				"path to frsdk.cfg. Usually something like C:\\FVSDK_8_9_5\\etc\\frsdk.cfg")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: fvsdk-enrol-all-videos [options]" << std::endl;
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
	Loggers->getLogger("facerecognition").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	if (!boost::filesystem::exists(inputDirectoryVideos)) {
		appLogger.error("The given input video directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}

	// Read the video detections metadata (eyes, face-coords):
	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(landmarksMetadata.string());
		boost::archive::text_iarchive ia(ifs); // binary_iarchive
		ia >> pascVideoDetections;
	} // archive and stream closed when destructors are called


	// Read the sigset:
	auto videoSigset = facerecognition::utils::readPascSigset(sigsetFile, true);

	// Create the output directory if it doesn't exist yet:
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	
	auto faceRecEngine = std::make_shared<facerecognition::FaceVacsEngine>(fvsdkConfig, outputPath);

	ThreadPool threadPool(numThreads);
	vector<std::future<void>> threads;

	int counter = 0;
	for (auto& video : videoSigset)
	{
		++counter;
		appLogger.info("Enroling video " + std::to_string(counter) + " of " + std::to_string(videoSigset.size()) + "...");
		auto videoName = inputDirectoryVideos / video.dataPath;
		if (!boost::filesystem::exists(videoName)) {
			appLogger.info("Found a video in the query sigset that doesn't exist in the filesystem. Skipping it.");
			continue; // We have 5 videos in the video-training-sigset that don't exist in the database
		}

		path firPath = outputPath / video.dataPath;
		firPath.replace_extension(".fir");
		threads.emplace_back(threadPool.enqueue(enrolImageSet, videoName, faceRecEngine, firPath));
		//enrolImageSet(frames, faceRecEngine, firPath);

		/*
		firs.emplace_back(threadPool.enqueue([&faceRecEngine](const cv::Mat& frame, path firPath) { return faceRecEngine.createFir(facerecognition::matToFRsdkImage(frame), firPath); }, frames[frameNum], firPath));
		*/

	}
	// Wait until all frames of this video are enroled:
	for (auto& t : threads) {
		t.get();
	}
	// Note: If a frame can't be enroled, the FIR file will just be missing.
	return EXIT_SUCCESS;
}
