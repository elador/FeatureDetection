/*
 * train-frame-extract-nnet.cpp
 *
 *  Created on: 11.09.2014
 *      Author: Patrik Huber
 *
 * Example:
 * train-frame-extract-nnet ...
 *   
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

#include "tiny_cnn.h"

#include "imageio/LandmarkMapper.hpp"
#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/PascVideoEyesLandmarkFormatParser.hpp"

#include "logging/LoggerFactory.hpp"

using namespace imageio;
namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
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
using std::shared_ptr;
using std::vector;
using std::string;

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"
#include "boost/archive/binary_iarchive.hpp"
#include "boost/serialization/optional.hpp"
#include "boost/serialization/vector.hpp"
#include <iostream>
class PascVideoDetection
{
public:
	static PascVideoDetection readFromCsv(std::string line);
private:
	friend class boost::serialization::access;
	// When the class Archive corresponds to an output archive, the
	// & operator is defined similar to <<.  Likewise, when the class Archive
	// is a type of input archive the & operator is defined similar to >>.
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & frame_id;
		ar & fcen_x;
		ar & fcen_y;
		ar & fwidth;
		ar & fheight;
		ar & fpose_y;
		ar & re_x;
		ar & re_y;
		ar & le_x;
		ar & le_y;
	}
public:
	// The class members have the same name as in the header line in the csv file
	std::string frame_id;
	int fcen_x;
	int fcen_y;
	int fwidth;
	int fheight;
	float fpose_y; // yaw
	boost::optional<int> re_x; // eye coordinates may or may not be present
	boost::optional<int> re_y; // re = which eye? document here.
	boost::optional<int> le_x;
	boost::optional<int> le_y;
};

std::vector<PascVideoDetection> readPascVideoDetections(boost::filesystem::path csvFile)
{
	std::vector<PascVideoDetection> detections;

	std::ifstream file(csvFile.string());
	string line;
	std::getline(file, line); // The header line, we skip it

	while (std::getline(file, line))
	{
		PascVideoDetection detection;

		vector<string> tokens;
		boost::trim_right_if(line, boost::is_any_of("\r")); // Windows line-endings are \r\n, Linux only \n. Thus, when a file has been created on windows and is read on linux, we need to remove the trailing \r.
		boost::split(tokens, line, boost::is_any_of(","));
		
		detection.frame_id = tokens[0];
		detection.fcen_x = lexical_cast<float>(tokens[1]);
		detection.fcen_y = lexical_cast<float>(tokens[2]);
		detection.fwidth = lexical_cast<float>(tokens[3]);
		detection.fheight = lexical_cast<float>(tokens[4]);
		detection.fpose_y = lexical_cast<float>(tokens[5]);
		if (tokens[6] != "") {
			detection.le_x = lexical_cast<float>(tokens[6]);
		}
		if (tokens[7] != "") {
			detection.le_y = lexical_cast<float>(tokens[7]);
		}
		if (tokens[8] != "") {
			detection.re_x = lexical_cast<float>(tokens[8]);
		}
		if (tokens[9] != "") {
			detection.re_y = lexical_cast<float>(tokens[9]);
		}
		detections.emplace_back(detection);
	}

	return detections;
}

vector<Mat> getFrames(path videoFilename)
{
	vector<Mat> frames;

	cv::VideoCapture cap(videoFilename.string());
	if (!cap.isOpened())
		throw("Couldn't open video file.");

	Mat img;
	while (cap.read(img)) {
		frames.emplace_back(img);
	}

	return frames;
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
	path inputDirectory;
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
			("input,i", po::value<path>(&inputDirectory)->required(),
				"input folder with training videos")
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
			cout << "Usage: train-frame-extract-nnet [options]\n";
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
	Loggers->getLogger("train-frame-extract-nnet").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("train-frame-extract-nnet");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	vector<PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(R"(C:\Users\Patrik\Documents\GitHub\data\PaSC\pasc_training_video_pittpatt_detection.txt)"); // ("pasc.bin", std::ios::binary | std::ios::in)
		boost::archive::text_iarchive ia(ifs); // binary_iarchive
		ia >> pascVideoDetections;
	} // archive and stream closed when destructors are called

	// Read the PaSC video landmarks:
	shared_ptr<imageio::NamedLandmarkSource> landmarkSource;
	vector<path> groundtruthDirs{ inputLandmarks };
	if (boost::iequals(landmarkType, "PaSC-video-PittPatt-detections")) {
		landmarkSource = make_shared<imageio::DefaultNamedLandmarkSource>(imageio::LandmarkFileGatherer::gather(nullptr, ".csv", imageio::GatherMethod::SEPARATE_FILES, groundtruthDirs), make_shared<imageio::PascVideoEyesLandmarkFormatParser>());
	}
	else {
		appLogger.error("Invalid landmarks type.");
		return EXIT_FAILURE;
	}

	// Create the output directory if it doesn't exist yet:
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	
	// Read all videos:
	vector<path> trainingVideos;
	copy(boost::filesystem::directory_iterator(inputDirectory), boost::filesystem::directory_iterator(), back_inserter(trainingVideos));
	std::random_device rd;
	auto videosSeed = rd();
	auto framesSeed = rd();
	std::mt19937 rndGenVideos(videosSeed);
	std::mt19937 rndGenFrames(framesSeed);
	std::uniform_int_distribution<> rndVidDistr(0, trainingVideos.size() - 1);
	auto randomVideo = std::bind(rndVidDistr, rndGenVideos);
	
	// Select random subset of videos:
	int numVideosToTrain = 2;
	int numFramesPerVideo = 2;
	for (int i = 0; i < numVideosToTrain; ++i) {
		auto videoFilename = trainingVideos[randomVideo()];
		auto frames = getFrames(videoFilename);
		// Select random subset of frames:
		std::uniform_int_distribution<> rndFrameDistr(0, frames.size() - 1);
		for (int j = 0; j < numFramesPerVideo; ++j) {
			int frameNum = rndFrameDistr(rndGenFrames);
			auto frame = frames[frameNum];
			// Get the landmarks for this frame:
			frameNum = frameNum + 1; // frame numbering in the CSV starts with 1
			std::ostringstream ss;
			ss << std::setw(3) << std::setfill('0') << frameNum;
			string frameName = videoFilename.stem().string() + "/" + videoFilename.stem().string() + "-" + ss.str() + ".jpg";
			//imageio::LandmarkCollection landmarks = landmarkSource->get(frameName);
			// Use facebox (later: or eyes) to run the engine
			/*auto result1 = std::find(std::begin(landmarks.getLandmarks()), std::end(landmarks.getLandmarks()), [](const shared_ptr<Landmark>& l) { return (l->getName() == "face"); });
			if (result1 != std::end(landmarks.getLandmarks())) {
				
			}
			else {
				continue;
			}*/
		}
	}

	
	// resulting score = label, facebox = input, resize it
	// Train NN

	return EXIT_SUCCESS;
}
