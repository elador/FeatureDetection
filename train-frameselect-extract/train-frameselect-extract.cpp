/*
 * train-frame-extract-nnet.cpp
 *
 *  Created on: 11.09.2014
 *      Author: Patrik Huber
 *
 * Ideally we'd use video, match against highres stills? (and not the lowres). Because if still are lowres/bad, we could match a
 * good frame against a bad gallery, which would give a bad score, but it shouldn't, because the frame is good.
 * Do we have labels for this?
 * Maybe "sensor_id","stage_id","env_id","illuminant_id" in the files emailed by Ross.
 *
 * Example:
 * train-frame-extract-nnet ...
 *   
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
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
#include "boost/archive/binary_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"

#include <boost/timer.hpp>
#include <boost/progress.hpp>

#include <frsdk/config.h>
#include <frsdk/enroll.h>
#include <frsdk/match.h>
#include <frsdk/eyes.h>

#include "imageio/MatSerialization.hpp"
#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"

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

// for the CoutEnrol
std::ostream&
operator<<(std::ostream& o, const FRsdk::Position& p)
{
	o << "[" << p.x() << ", " << p.y() << "]";
	return o;
}

// templates instead of inheritance? (compile-time fine, don't need runtime selection)
class FaceVacsEngine
{
public:
	// tempDirectory a path where temp files stored... only FIRs? More? firDir?
	FaceVacsEngine(path frsdkConfig, path tempDirectory) : tempDir(tempDirectory) {
		// Todo: Create the tempDir if it doesn't exist yet

		// initialize and resource allocation
		cfg = std::make_unique<FRsdk::Configuration>(frsdkConfig.string());
		firBuilder = std::make_unique<FRsdk::FIRBuilder>(*cfg.get());
		// initialize matching facility
		me = std::make_unique<FRsdk::FacialMatchingEngine>(*cfg.get());
		// load the fir population (gallery)
		population = std::make_unique<FRsdk::Population>(*cfg.get());

		// Add somewhere: 
		//catch (const FRsdk::FeatureDisabled& e) 
		//catch (const FRsdk::LicenseSignatureMismatch& e)
	};

	// because it often stays the same
	void enrollGallery(std::vector<facerecognition::FaceRecord> galleryRecords, path databasePath)
	{
		this->galleryRecords = galleryRecords;

		// We first enroll the whole gallery:
		cout << "Loading the input images..." << endl;
		// create an enrollment processor
		FRsdk::Enrollment::Processor proc(*cfg);
		// Todo: FaceVACS should be able to do batch-enrolment?
		auto cnt = 0;
		for (auto& r : galleryRecords) {
			++cnt;
			std::cout << cnt << std::endl;
			// Does the FIR already exist in the temp-dir? If yes, skip FD/EyeDet!
			auto firPath = tempDir / r.dataPath;
			firPath.replace_extension(".fir");
			if (boost::filesystem::exists(firPath)) {
				continue;
			}

			boost::optional<path> fir = createFir(databasePath / r.dataPath);
			if (!fir) {
				continue;
			}
		}
		// Enrol all the FIRs now:
		for (auto& r : galleryRecords) {
			auto firPath = tempDir / r.dataPath;
			firPath.replace_extension(".fir");
			std::ifstream firIn(firPath.string(), std::ios::in | std::ios::binary);
			// Some of the FIRs might not exist due to enrolment failure
			//std::cout << firPath.string() << std::endl;
			if (firIn.is_open() && firIn.good()) {
				try {
					auto fir = firBuilder->build(firIn);
					population->append(fir, firPath.string());
				}
				catch (const FRsdk::FeatureDisabled& e) {
					std::cout << e.what() << std::endl;
				}
				catch (const FRsdk::LicenseSignatureMismatch& e) {
					std::cout << e.what() << std::endl;
				}
				catch (std::exception& e) {
					std::cout << e.what() << std::endl;
				}
			}
		}
	};

	// Make a matchAgainstGallery or matchSingle function as well
	// matches a pair of images, Cog LMs
	double match(cv::Mat first, cv::Mat second)
	{
		return 0.0;
	};
	// Always pass in images/image-filenames. The "outside" should never have to worry
	// about FIRs or other intermediate representations.
	// Eyes: Currently unused, use PaSC-tr eyes in the future
	double match(path g, std::string gallerySubjectId, cv::Mat p, std::string probeFilename, cv::Vec2f firstEye, cv::Vec2f secondEye)
	{
		auto galleryFirPath = tempDir / g.filename();
		galleryFirPath.replace_extension(".fir"); // we probably don't need this - only the basename / subject ID (if it is already enroled)

		auto tempProbeImageFilename = tempDir / path(probeFilename).filename();
		cv::imwrite(tempProbeImageFilename.string(), p);

		boost::optional<path> probeFrameFir = createFir(tempProbeImageFilename);
		if (!probeFrameFir) {
			std::cout << "Couldn't enroll the probe - not a good frame. Return score 0.0." << std::endl;
			return 0.0; // Couldn't enroll the probe - not a good frame. Return score 0.0.
		}


		std::ifstream firStream(probeFrameFir->string(), std::ios::in | std::ios::binary);
		FRsdk::FIR fir = firBuilder->build(firStream);

		//compare() does not care about the configured number of Threads
		//for the comparison algorithm. It uses always one thrad to
		//compare all inorder to preserve the order of the scores
		//according to the order in the population (orer of adding FIRs to
		//the population)
		FRsdk::CountedPtr<FRsdk::Scores> scores = me->compare(fir, *population);

		// Find the score from "g" in the gallery
		auto subjectInGalleryIter = std::find_if(begin(galleryRecords), end(galleryRecords), [gallerySubjectId](const facerecognition::FaceRecord& g) { return (g.subjectId == gallerySubjectId); });
		if (subjectInGalleryIter == end(galleryRecords)) {
			return 0.0; // We didn't find the given gallery image / subject ID in the enroled gallery. Throw?
		}
		auto subjectInGalleryIdx = std::distance(begin(galleryRecords), subjectInGalleryIter);
		auto givenProbeVsGivenGallery = std::next(begin(*scores), subjectInGalleryIdx);
		return *givenProbeVsGivenGallery;
	};
	// matches a pair of images, given LMs as Cog init
	// match(fn, fn, ...)
	// match(Mat, Mat, ...)

	// creates the FIR in temp-dir. Returns path to FIR if successful.
	// Todo: Directly return FIR?
	boost::optional<path> createFir(path image)
	{
		// create an enrollment processor
		FRsdk::Enrollment::Processor proc(*cfg);
		// Todo: FaceVACS should be able to do batch-enrolment?
		
		// Does the FIR already exist in the temp-dir? If yes, skip FD/EyeDet!
		auto firPath = tempDir / image.filename();
		firPath.replace_extension(".fir");
		if (boost::filesystem::exists(firPath)) {
			//throw std::runtime_error("Unexpected. Should check for that.");
			// just overwrite
		}

		FRsdk::Image img(FRsdk::ImageIO::load(image.string()));
		FRsdk::SampleSet enrollmentImages;
		auto sample = FRsdk::Sample(img);
		// Once we have pasc-still-training landmarks from Ross, we can use those here. In the mean-time, use the Cog Eyefinder:
		FRsdk::Face::Finder faceFinder(*cfg);
		FRsdk::Eyes::Finder eyesFinder(*cfg);
		float mindist = 0.01f; // minEyeDist, def = 0.1
		float maxdist = 0.3f; // maxEyeDist, def = 0.4
		FRsdk::Face::LocationSet faceLocations = faceFinder.find(img, mindist, maxdist);
		if (faceLocations.empty()) {
			std::cout << "FaceFinder: No face found." << std::endl;
			return boost::none;
		}
		// We just use the first found face:
		// doing eyes finding
		FRsdk::Eyes::LocationSet eyesLocations = eyesFinder.find(img, *faceLocations.begin());
		if (eyesLocations.empty()) {
			std::cout << "EyeFinder: No eyes found." << std::endl;
			return boost::none;
		}
		auto foundEyes = eyesLocations.begin(); // We just use the first found eyes
		auto firstEye = FRsdk::Position(foundEyes->first.x(), foundEyes->first.y()); // first eye (image left-most), second eye (image right-most)
		auto secondEye = FRsdk::Position(foundEyes->second.x(), foundEyes->second.y());
		sample.annotate(FRsdk::Eyes::Location(firstEye, secondEye, foundEyes->firstConfidence, foundEyes->secondConfidence));
		enrollmentImages.push_back(sample);
		// create the needed interaction instances
		FRsdk::Enrollment::Feedback feedback(new EnrolCoutFeedback(firPath.string()));
		// Note: We could do a SignalBasedFeedback, that sends a signal. We could then here wait for the signal and process stuff, i.e. get the FIR directly.
		// do the enrollment
		proc.process(enrollmentImages.begin(), enrollmentImages.end(), feedback);
		// Better solution would be to change the callback into some event-based stuff
		if (boost::filesystem::exists(firPath)) {
			return firPath;
		}
		else {
			return boost::none;
		}
	};

private:
	path tempDir;
	std::unique_ptr<FRsdk::Configuration> cfg; // static? (recommendation by fvsdk doc)
	std::unique_ptr<FRsdk::FIRBuilder> firBuilder;
	std::unique_ptr<FRsdk::FacialMatchingEngine> me;
	std::vector<facerecognition::FaceRecord> galleryRecords; // We keep this for the subject IDs
	std::unique_ptr<FRsdk::Population> population; // enrolled gallery FIRs, same order

	class EnrolCoutFeedback : public FRsdk::Enrollment::FeedbackBody
	{
	public:
		EnrolCoutFeedback(const std::string& firFilename)
			: firFN(firFilename), firvalid(false) { }
		~EnrolCoutFeedback() {}

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

};

// Caution: This will eat a lot of RAM, 1-2 GB for 600 RGB frames at 720p
vector<Mat> getFrames(path videoFilename)
{
	vector<Mat> frames;

	cv::VideoCapture cap(videoFilename.string());
	if (!cap.isOpened())
		throw("Couldn't open video file.");

	Mat img;
	while (cap.read(img)) {
		frames.emplace_back(img.clone()); // we need to clone, otherwise we'd just get a reference to the same 'img' instance
	}

	return frames;
}

// pascFrameNumber starts with 1. Your counting might start with 0, so add 1 to it before passing it here.
std::string getPascFrameName(path videoFilename, int pascFrameNumber)
{
	std::ostringstream ss;
	ss << std::setw(3) << std::setfill('0') << pascFrameNumber;
	return videoFilename.stem().string() + "/" + videoFilename.stem().string() + "-" + ss.str() + ".jpg";
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path inputDirectoryVideos, inputDirectoryStills;
	path inputLandmarks;
	path outputPath;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("input-videos,i", po::value<path>(&inputDirectoryVideos)->required(),
				"input folder with training videos")
			("input-stills,j", po::value<path>(&inputDirectoryStills)->required(),
				"input folder with training videos")
			("landmarks,l", po::value<path>(&inputLandmarks)->required(),
				"input landmarks in boost::serialization text format")
			("output,o", po::value<path>(&outputPath)->default_value("."),
				"path to an output folder")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: train-frameselect-extract [options]" << std::endl;
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

	// Read the video detections metadata (eyes, face-coords):
	vector<facerecognition::PascVideoDetection> pascVideoDetections;
	{
		std::ifstream ifs(inputLandmarks.string()); // ("pasc.bin", std::ios::binary | std::ios::in)
		boost::archive::text_iarchive ia(ifs); // binary_iarchive
		ia >> pascVideoDetections;
	} // archive and stream closed when destructors are called

	// We would read the still images detection metadata (eyes, face-coords) here, but it's not provided with PaSC. Emailed Ross.
	// TODO!

	// Read the training-video xml sigset and the training-still sigset to get the subject-id metadata:
	auto videoQuerySet = facerecognition::utils::readPascSigset(R"(C:\Users\Patrik\Documents\GitHub\data\PaSC\Web\nd1Fall2010VideoPaSCTrainingSet.xml)", true);
	auto stillTargetSet = facerecognition::utils::readPascSigset(R"(C:\Users\Patrik\Documents\GitHub\data\PaSC\Web\nd1Fall2010PaSCamerasStillTrainingSet.xml)", true);

	// Create the output directory if it doesn't exist yet:
	if (!boost::filesystem::exists(outputPath)) {
		boost::filesystem::create_directory(outputPath);
	}
	
	// Read all videos:
	if (!boost::filesystem::exists(inputDirectoryVideos)) {
		appLogger.error("The given input files directory doesn't exist. Aborting.");
		return EXIT_FAILURE;
	}


	cv::Mat testbig = cv::Mat::ones(100, 150, CV_8UC2);
	cv::Rect roi(10, 12, 30, 34);
	Mat test = testbig(roi);
	test.setTo(100.0f);
	cout << test.isContinuous();
	std::ofstream ofPascT("test.txt");
	{ // use scope to ensure archive goes out of scope before stream
		boost::archive::text_oarchive oa(ofPascT);
		oa << test;
	}
	ofPascT.close();

	Mat test2;
	std::ifstream ifPascT("test.txt");
	{ // use scope to ensure archive goes out of scope before stream
		boost::archive::text_iarchive ia(ifPascT);
		ia >> test2;
	}
	ifPascT.close();

	/*
	vector<path> trainingVideos;
	try {
		copy(boost::filesystem::directory_iterator(inputDirectoryVideos), boost::filesystem::directory_iterator(), back_inserter(trainingVideos));
	}
	catch (boost::filesystem::filesystem_error& e) {
		string errorMsg("Error while loading the video files from the given input directory: " + string(e.what()));
		appLogger.error(errorMsg);
		return EXIT_FAILURE;
	}*/

	FaceVacsEngine faceRecEngine(R"(C:\FVSDK_8_9_5\etc\frsdk.cfg)", R"(C:\Users\Patrik\Documents\GitHub\aaatmp)");
	//auto stillTargetSetSmall = { stillTargetSet[1092]/*, stillTargetSet[0] */};
	//path = L"C:\\Users\\Patrik\\Documents\\GitHub\\aaatmp\\06340d96.fir"
	//auto dp = path("06340d96.jpg");
	//auto galSubj = std::find_if(begin(stillTargetSet), end(stillTargetSet), [dp](const facerecognition::FaceRecord& g) { return (g.dataPath == dp); });
	// check
	//auto galSubjIdx = std::distance(begin(stillTargetSet), galSubj);
	stillTargetSet.resize(250); // 1000 = FIR limit atm
	faceRecEngine.enrollGallery(stillTargetSet, inputDirectoryStills);

	std::random_device rd;
	auto videosSeed = rd();
	auto framesSeed = rd();
	auto posTargetsSeed = rd();
	auto negTargetsSeed = rd();
	std::mt19937 rndGenVideos(videosSeed);
	std::mt19937 rndGenFrames(framesSeed);
	std::mt19937 rndGenPosTargets(posTargetsSeed);
	std::mt19937 rndGenNegTargets(negTargetsSeed);
	std::uniform_int_distribution<> rndVidDistr(0, videoQuerySet.size() - 1);
	auto randomVideo = std::bind(rndVidDistr, rndGenVideos);
	
	// The training data:
	vector<Mat> trainingFrames;
	vector<float> labels; // the score difference to the value we would optimally like
						  // I.e. if it's a positive pair, the label is the difference to 1.0
						  // In case of a negative pair, the label is the difference to 0.0

	// Select random subset of videos: (Note: This means we may select the same video twice - not so optimal?)
	int numVideosToTrain = 2;
	int numFramesPerVideo = 2;
	int numPositivePairsPerFrame = 1;
	int numNegativePairsPerFrame = 0;
	for (int i = 0; i < numVideosToTrain; ++i) {
		numPositivePairsPerFrame = 1; // it has to be set to 1 again if we set it to 0 below in the previous iteration.
		auto queryVideo = videoQuerySet[randomVideo()];
		if (!boost::filesystem::exists(inputDirectoryVideos / queryVideo.dataPath)) {  // Shouldn't be necessary, but there are 5 videos in the xml sigset that we don't have.
			continue;
		}
		auto frames = getFrames(inputDirectoryVideos / queryVideo.dataPath);
		// For the currently selected video, partition the target set. The distributions don't change each frame, whole video has the same FaceRecord.
		auto bound = std::partition(begin(stillTargetSet), end(stillTargetSet), [queryVideo](facerecognition::FaceRecord& target) { return target.subjectId == queryVideo.subjectId; });
		// begin to bound = positive pairs, rest = negative
		auto numPositivePairs = std::distance(begin(stillTargetSet), bound);
		auto numNegativePairs = std::distance(bound, end(stillTargetSet));
		if (numPositivePairs == 0) { // Not sure if this should be happening in the PaSC sigsets - ask Ross
			numPositivePairs = 1; // Causes the uniform_int_distribution c'tor not to crash because of negative value
			numPositivePairsPerFrame = 0; // Causes the positive-pairs stuff further down to be skipped
			continue; // only for debugging!
		}
		std::uniform_int_distribution<> rndPositiveDistr(0, numPositivePairs - 1); // -1 because all vector indices start with 0
		std::uniform_int_distribution<> rndNegativeDistr(numPositivePairs, stillTargetSet.size() - 1);
		// Select random subset of frames:
		std::uniform_int_distribution<> rndFrameDistr(0, frames.size() - 1);
		for (int j = 0; j < numFramesPerVideo; ++j) {
			int frameNum = rndFrameDistr(rndGenFrames);
			auto frame = frames[frameNum];
			// Get the landmarks for this frame:
			string frameName = getPascFrameName(queryVideo.dataPath, frameNum + 1);
			std::cout << "=== STARTING TO PROCESS " << frameName << " ===" << std::endl;
			auto landmarks = std::find_if(begin(pascVideoDetections), end(pascVideoDetections), [frameName](const facerecognition::PascVideoDetection& d) { return (d.frame_id == frameName); });
			// Use facebox (later: or eyes) to run the engine
			if (landmarks == std::end(pascVideoDetections)) {
				appLogger.info("Chose a frame but could not find a corresponding entry in the metadata file - skipping it.");
				continue; // instead, do a while-loop and count the number of frames with landmarks (so we don't skip videos if we draw bad values)
				// We throw away the frames with no landmarks. This basically means our algorithm will only be trained on frames where PittPatt succeeds, and
				// frames where it doesn't are unknown data to our nnet. I think we should try including these frames as well, e.g. with an error/label of 1.0.
			}
			// Choose one random positive and one random negative pair (we could use more, this is just the first attempt):
			// Actually with the Cog engine we could enrol the whole gallery and get all the scores in one go, should be much faster
			for (int k = 0; k < numPositivePairsPerFrame; ++k) { // we can also get twice (or more) times the same, but doesn't matter for now
				auto targetStill = stillTargetSet[rndPositiveDistr(rndGenPosTargets)];
				// TODO get LMs for targetStill from PaSC - see further up, email Ross
				// match (targetStill (FaceRecord), LMS TODO ) against ('frame' (Mat), queryVideo (FaceRecord), landmarks)
				double recognitionScore = faceRecEngine.match(inputDirectoryStills / targetStill.dataPath, targetStill.subjectId, frame, frameName, cv::Vec2f(), cv::Vec2f());
				std::cout << "===== Got a score!" << recognitionScore << "=====" << std::endl;
				string out = (outputPath / path(frameName).stem()).string() + "_" + std::to_string(recognitionScore) + ".png";
				cv::imwrite(out, frame);
				double frameScore = recognitionScore; // This is our label. It's a positive pair, so the higher the score the higher our framescore.

				// The face box is always given. Only the eyes are missing sometimes.
				cv::Rect roi(landmarks->fcen_x - landmarks->fwidth / 2.0 - landmarks->fwidth / 10.0, landmarks->fcen_y - landmarks->fheight / 2.0 - landmarks->fheight / 10.0, landmarks->fwidth + landmarks->fwidth / 5.0, landmarks->fheight + landmarks->fheight / 5.0);
				Mat croppedFace = frame(roi);
				cv::resize(croppedFace, croppedFace, cv::Size(32, 32)); // need to set this higher... 40x40? What about aspect ratio?
				cv::cvtColor(croppedFace, croppedFace, cv::COLOR_BGR2GRAY);

				trainingFrames.emplace_back(croppedFace);
				labels.emplace_back(frameScore);
			}
			for (int k = 0; k < numNegativePairsPerFrame; ++k) {
				double recognitionScore;
				double frameScore = 1.0 - recognitionScore; // This is our label. It's a negative pair, so a facerec-score close to 0 is good, meaning 1-frScore is the frameScore
				// This is problematic here because we return a score of 0 for frames that we couldn't enrol - they would get 0 error here, which is bad.
			}

			// Run the engine:
			// in: frame, eye/face coords, plus one positive image from the still-sigset with its eye/face coords
			// out: score
			// Later: Include several positive scores, and also negative pairs
			// I.e. enroll the whole gallery once, then match the query frame and get all scores?

			// From this pair:
			// resulting score difference to class label = label, facebox = input, resize it
			//trainingFrames.emplace_back(frame); // store only the resized frames!
			//labels.emplace_back(1.0f); // 1.0 means good frame, 0.0 means bad frame.
		}
	}

	// Engine:
	// libFaceRecognition: CMake-Option for dependency on FaceVACS
	// Class Engine;
	// FaceVacsExecutableRunner : Engine; C'tor: Path to directory with binaries or individual binaries?
	// FaceVacs : Engine; C'tor - nothing? FV-Config?

	// Or: All the FaceVacs stuff in libFaceVacsWrapper. Direct-calls and exe-runner. With CMake-Option.

	return EXIT_SUCCESS;
}
