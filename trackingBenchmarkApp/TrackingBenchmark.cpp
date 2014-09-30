/*
 * TrackingBenchmark.cpp
 *
 *  Created on: 17.02.2014
 *      Author: poschmann
 */

#include "TrackingBenchmark.hpp"
#include "logging/LoggerFactory.hpp"
#include "logging/ConsoleAppender.hpp"
#include "logging/FileAppender.hpp"
#include "imageio/VideoImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/BobotAnnotationSource.hpp"
#include "imageio/SimpleAnnotationSource.hpp"
#include "imageio/BobotAnnotationSink.hpp"
#include "imageio/SimpleAnnotationSink.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/AgeBasedExampleManagement.hpp"
#include "classification/ConfidenceBasedExampleManagement.hpp"
#include "classification/UnlimitedExampleManagement.hpp"
#include "classification/FixedTrainableProbabilisticSvmClassifier.hpp"
#include "libsvm/LibSvmClassifier.hpp"
#include "condensation/ConstantVelocityModel.hpp"
#include "condensation/OpticalFlowTransitionModel.hpp"
#include "condensation/LowVarianceSampling.hpp"
#include "condensation/ExtendedHogBasedMeasurementModel.hpp"
#include "condensation/FilteringStateExtractor.hpp"
#include "condensation/WeightedMeanStateExtractor.hpp"
#include "condensation/Sample.hpp"
#include "boost/program_options.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/optional.hpp"
#include "boost/lexical_cast.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <sstream>
#include <algorithm>

using namespace logging;
using namespace classification;
using namespace std::chrono;
using libsvm::LibSvmClassifier;
using cv::Point;
using cv::Rect;
using boost::property_tree::info_parser::read_info;
using boost::lexical_cast;
using std::milli;
using std::move;
using std::ofstream;
using std::ostringstream;
using std::istringstream;
using std::runtime_error;
using std::invalid_argument;

TrackingBenchmark::TrackingBenchmark(ptree& config) {
	initTracking(config);
}

unique_ptr<ExampleManagement> TrackingBenchmark::createExampleManagement(ptree& config, shared_ptr<BinaryClassifier> classifier, bool positive) {
	if (config.get_value<string>() == "unlimited") {
		return unique_ptr<ExampleManagement>(new UnlimitedExampleManagement(config.get<size_t>("required")));
	} else if (config.get_value<string>() == "agebased") {
		return unique_ptr<ExampleManagement>(new AgeBasedExampleManagement(config.get<size_t>("capacity"), config.get<size_t>("required")));
	} else if (config.get_value<string>() == "confidencebased") {
		return unique_ptr<ExampleManagement>(new ConfidenceBasedExampleManagement(classifier, positive, config.get<size_t>("capacity"), config.get<size_t>("required")));
	} else {
		throw invalid_argument("invalid example management type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableProbabilisticSvmClassifier> TrackingBenchmark::createSvm(ptree& config) {
	shared_ptr<LibSvmClassifier> svm = make_shared<LibSvmClassifier>(make_shared<LinearKernel>(), config.get<double>("training.C"));
	svm->setPositiveExampleManagement(
			unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("training.positiveExamples"), svm, true)));
	svm->setNegativeExampleManagement(
			unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("training.negativeExamples"), svm, false)));
	return make_shared<FixedTrainableProbabilisticSvmClassifier>(svm,
			config.get<double>("probabilistic.logisticA"), config.get<double>("probabilistic.logisticB"));
}

void TrackingBenchmark::initTracking(ptree& config) {
	// create measurement model
	shared_ptr<TrainableProbabilisticSvmClassifier> classifier = createSvm(config.get_child("measurement.classifier"));
	shared_ptr<ExtendedHogBasedMeasurementModel> measurementModel = make_shared<ExtendedHogBasedMeasurementModel>(classifier);
	measurementModel->setHogParams(
			config.get<size_t>("measurement.cellSize"),
			config.get<size_t>("measurement.cellCount"),
			config.get<bool>("measurement.signedAndUnsigned"),
			config.get<bool>("measurement.interpolateBins"),
			config.get<bool>("measurement.interpolateCells"),
			config.get<int>("measurement.octaveLayerCount"));
	measurementModel->setRejectionThreshold(
			config.get<double>("measurement.rejectionThreshold"));
	measurementModel->setUseSlidingWindow(
			config.get<bool>("measurement.useSlidingWindow"),
			config.get<bool>("measurement.conservativeReInit"));
	measurementModel->setNegativeExampleParams(
			config.get<size_t>("measurement.negativeExampleCount"),
			config.get<size_t>("measurement.initialNegativeExampleCount"),
			config.get<size_t>("measurement.randomExampleCount"),
			config.get<float>("measurement.negativeScoreThreshold"));
	measurementModel->setOverlapThresholds(
			config.get<double>("measurement.positiveOverlapThreshold"),
			config.get<double>("measurement.negativeOverlapThreshold"));
	ExtendedHogBasedMeasurementModel::Adaptation adaptation;
	if (config.get<string>("measurement.adaptation") == "NONE")
		adaptation = ExtendedHogBasedMeasurementModel::Adaptation::NONE;
	else if (config.get<string>("measurement.adaptation") == "POSITION")
		adaptation = ExtendedHogBasedMeasurementModel::Adaptation::POSITION;
	else if (config.get<string>("measurement.adaptation") == "TRAJECTORY")
		adaptation = ExtendedHogBasedMeasurementModel::Adaptation::TRAJECTORY;
	else if (config.get<string>("measurement.adaptation") == "CORRECTED_TRAJECTORY")
		adaptation = ExtendedHogBasedMeasurementModel::Adaptation::CORRECTED_TRAJECTORY;
	else
		throw invalid_argument("invalid adaptation type: " + config.get<string>("measurement.adaptation"));
	measurementModel->setAdaptation(adaptation,
			config.get<double>("measurement.adaptationThreshold"),
			config.get<double>("measurement.exclusionThreshold"));

	// create transition model
	shared_ptr<TransitionModel> transitionModel;
	if (config.get<string>("transition") == "constantVelocity") {
		transitionModel = make_shared<ConstantVelocityModel>(
				config.get<double>("transition.positionDeviation"), config.get<double>("transition.sizeDeviation"));
	} else if (config.get<string>("transition") == "opticalFlow") {
		shared_ptr<TransitionModel> fallbackModel = make_shared<ConstantVelocityModel>(
				config.get<double>("transition.fallback.positionDeviation"), config.get<double>("transition.fallback.sizeDeviation"));
		transitionModel = make_shared<OpticalFlowTransitionModel>(
				fallbackModel, config.get<double>("transition.positionDeviation"), config.get<double>("transition.sizeDeviation"));
	} else {
		throw invalid_argument("invalid transition model type: " + config.get<string>("transition"));
	}

	// create tracker
	shared_ptr<StateExtractor> stateExtractor = make_shared<FilteringStateExtractor>(make_shared<WeightedMeanStateExtractor>());
	tracker = unique_ptr<CondensationTracker>(new CondensationTracker(
			transitionModel, measurementModel, make_shared<LowVarianceSampling>(), stateExtractor,
			config.get<unsigned int>("resampling.particleCount"), config.get<double>("resampling.randomRate"),
			config.get<double>("resampling.minSize"), config.get<double>("resampling.maxSize")));
}

pair<double, double> TrackingBenchmark::runTest(shared_ptr<AnnotatedImageSource> imageSource, shared_ptr<AnnotationSink> annotationSink) {
	steady_clock::time_point start = steady_clock::now();
	duration<double> condensationTime;

	// initialization via ground truth
	if (!imageSource->next())
		throw runtime_error("there are no images in source");
	size_t frames = 1;
	Mat frame = imageSource->getImage();
	Rect truth = imageSource->getAnnotation();
	if (truth.area() > 0
			&& truth.x >= 0 && truth.y >= 0
			&& truth.br().x < frame.cols && truth.br().y < frame.rows) {
		steady_clock::time_point condensationStart = steady_clock::now();
		optional<Rect> position = tracker->initialize(frame, truth);
		if (!position)
			throw runtime_error("Adaptive tracker could not be initialized with " + lexical_cast<string>(truth));
		steady_clock::time_point condensationEnd = steady_clock::now();
		condensationTime += duration_cast<milliseconds>(condensationEnd - condensationStart);
		annotationSink->add(*position);
	}

	// adaptive tracking
	while (imageSource->next()) {
		frames++;
		frame = imageSource->getImage();
		steady_clock::time_point condensationStart = steady_clock::now();
		optional<Rect> position = tracker->process(frame);
		steady_clock::time_point condensationEnd = steady_clock::now();
		condensationTime += duration_cast<milliseconds>(condensationEnd - condensationStart);
		if (position)
			annotationSink->add(*position);
		else
			annotationSink->add(Rect());
	}
	tracker->reset();

	steady_clock::time_point end = steady_clock::now();
	duration<double> time = duration_cast<milliseconds>(end - start);
	double fps = frames / time.count();
	double condensationFps = frames / condensationTime.count();
	return std::make_pair(fps, condensationFps);
}

double TrackingBenchmark::runTests(const path& resultsDirectory, size_t count, const ptree& testConfig, Logger& log) {
	string testName = testConfig.get_value<string>();
	optional<string> imageDirectory = testConfig.get_optional<string>("directory");
	optional<string> videoFile = testConfig.get_optional<string>("file");
	optional<string> simpleFile = testConfig.get_optional<string>("simple");
	optional<string> bobotFile = testConfig.get_optional<string>("bobot");
	shared_ptr<ImageSource> imageSource;
	if (videoFile && !imageDirectory)
		imageSource = make_shared<VideoImageSource>(*videoFile);
	else if (!videoFile && imageDirectory)
		imageSource = make_shared<DirectoryImageSource>(*imageDirectory);
	else
		throw invalid_argument("either a video file or a directory must be given for test " + testName);
	bool bobot = false;
	if (simpleFile && !bobotFile)
		bobot = false;
	else if (!simpleFile && bobotFile)
		bobot = true;
	else
		throw invalid_argument("either a bobot or a simple ground truth file must be given for test " + testName);
	shared_ptr<AnnotationSource> annotationSource;
	shared_ptr<AnnotationSink> annotationSink;
	path groundTruthFile;
	if (bobot) {
		shared_ptr<BobotAnnotationSource> bobotAnnotationSource = make_shared<BobotAnnotationSource>(*bobotFile, imageSource);
		annotationSource = bobotAnnotationSource;
		annotationSink = make_shared<BobotAnnotationSink>(bobotAnnotationSource->getVideoFilename(), imageSource);
		groundTruthFile = path(*bobotFile);
	} else { // simple
		annotationSource = make_shared<SimpleAnnotationSource>(*simpleFile);
		annotationSink = make_shared<SimpleAnnotationSink>();
		groundTruthFile = path(*simpleFile);
	}
	shared_ptr<AnnotatedImageSource> source = make_shared<AnnotatedImageSource>(imageSource, annotationSource);

	log.info("=======");
	log.info("Running test " + testName + " " + (count == 1 ? "1 time" : (std::to_string(count) + " times")));
	path testDirectory(resultsDirectory.string() + "/" + testName);
	if (!exists(testDirectory))
		create_directory(testDirectory);
	// TODO Boost must be compiled with --std=C++11 in order for the following line to be linkable (Linux & GCC)
	// see http://boost.2283326.n4.nabble.com/Filesystem-problems-with-g-std-c-0x-td2639716.html for further information
//	copy_file(groundTruthFile, testDirectory / "groundtruth");
	size_t skipped = 0;
	pair<double, double> fpsSum(0, 0);
	for (size_t i = 0; i < count; ++i) {
		source->reset();
		string outputFilename = testDirectory.string() + "/run" + std::to_string(i);
		string learnedFilename = testDirectory.string() + "/learned" + std::to_string(i);
		if (exists(path(outputFilename))) {
			skipped++;
		} else {
			annotationSink->open(outputFilename);
			try {
				pair<double, double> fps = runTest(source, annotationSink);
				fpsSum.first += fps.first;
				fpsSum.second += fps.second;
			} catch (std::exception& exc) {
				log.error(string("exception on run " + std::to_string(i) + ": ") + exc.what());
			}
			annotationSink->close();
		}
	}
	if (skipped == count)
		log.info("skipped all tests (were already done)");
	else if (skipped == 1)
		log.info("skipped 1 test (was already done)");
	else if (skipped > 1)
		log.info("skipped " + std::to_string(skipped) + " tests (were already done)");

	double hitThreshold = 0.5;
	size_t runCount = 0;
	double hitM = 0;
	double hitS = 0;
	double overlapM = 0;
	double overlapS = 0;
	ostringstream hitDetails;
	hitDetails.setf(std::ios_base::fixed, std::ios_base::floatfield);
	hitDetails.precision(1);
	hitDetails << '(';
	ostringstream overlapDetails;
	overlapDetails.setf(std::ios_base::fixed, std::ios_base::floatfield);
	overlapDetails.precision(1);
	overlapDetails << '(';
	for (size_t i = 0; i < count; ++i) {
		string annotationOutputFilename(testDirectory.string() + "/run" + std::to_string(i));
		shared_ptr<AnnotationSource> annotationOutput;
		if (bobot) {
			annotationOutput = make_shared<BobotAnnotationSource>(annotationOutputFilename, imageSource);
			imageSource->reset();
			imageSource->next();
		} else { // simple
			annotationOutput = make_shared<SimpleAnnotationSource>(annotationOutputFilename);
		}
		annotationSource->reset();

		ofstream overlapOutput(testDirectory.string() + "/overlap" + std::to_string(i));
		overlapOutput.setf(std::ios_base::fixed, std::ios_base::floatfield);
		overlapOutput.precision(3);
		size_t frameCount = 0;
		size_t hitCount = 0;
		double overlapSum = 0;
		while (annotationSource->next()) {
			Rect truth = annotationSource->getAnnotation();
			double overlap = 0;
			if (annotationOutput->next()) {
				Rect output = annotationOutput->getAnnotation();
				overlap = computeOverlap(truth, output);
			}
			overlapOutput << overlap << '\n';
			frameCount++;
			if (overlap >= hitThreshold)
				hitCount++;
			overlapSum += overlap;
		}
		overlapOutput.close();
		double hitScore = static_cast<double>(hitCount) / frameCount;
		double averageOverlap = overlapSum / frameCount;
		if (i > 0)
			hitDetails << ',' << ' ';
		hitDetails << (100 * hitScore) << '%';
		if (i > 0)
			overlapDetails << ',' << ' ';
		overlapDetails << (100 * averageOverlap) << '%';

		runCount++;
		if (runCount == 1) {
			hitM = hitScore;
			hitS = 0;
			overlapM = averageOverlap;
			overlapS = 0;
		} else {
			double newHitM = hitM + (hitScore - hitM) / runCount;
			double newHitS = hitS + (hitScore - hitM) * (hitScore - newHitM);
			hitM = newHitM;
			hitS = newHitS;
			double newOverlapM = overlapM + (averageOverlap - overlapM) / runCount;
			double newOverlapS = overlapS + (averageOverlap - overlapM) * (averageOverlap - newOverlapM);
			overlapM = newOverlapM;
			overlapS = newOverlapS;
		}
	}
	hitDetails << ')';
	overlapDetails << ')';
	double hitScoreMean = hitM;
	double hitScoreDeviation = 0;
	double averageOverlapMean = overlapM;
	double averageOverlapDeviation = 0;
	if (runCount > 1) {
		hitScoreDeviation = std::sqrt(hitS / (runCount - 1));
		averageOverlapDeviation = std::sqrt(overlapS / (runCount - 1));
	}
	ostringstream hitOutput;
	hitOutput.setf(std::ios_base::fixed, std::ios_base::floatfield);
	hitOutput.precision(1);
	hitOutput << "hits: " << (100 * hitScoreMean) << "% / " << (100 * hitScoreDeviation) << "% " << hitDetails.str();
	log.info(hitOutput.str());
	ostringstream overlapOutput;
	overlapOutput.setf(std::ios_base::fixed, std::ios_base::floatfield);
	overlapOutput.precision(1);
	overlapOutput << "overlap: " << (100 * averageOverlapMean) << "% / " << (100 * averageOverlapDeviation) << "% " << overlapDetails.str();
	log.info(overlapOutput.str());
	ostringstream timeOutput;
	timeOutput.setf(std::ios_base::fixed, std::ios_base::floatfield);
	timeOutput.precision(1);
	timeOutput << "speed: " << (fpsSum.first / count) << " fps, " << (fpsSum.second / count) << " fps (condensation only)";
	log.info(timeOutput.str());
	return 100 * averageOverlapMean;
}

double TrackingBenchmark::computeOverlap(Rect a, Rect b) const {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	if (unionArea == 0)
		return 0;
	return intersectionArea / unionArea;
}

// returns a path to a non-existing file by adding a number to the given base pathname
path getNonExistingFile(path basename) {
	path file;
	size_t index = 0;
	do {
		file = path(basename.string() + std::to_string(index));
		index++;
	} while (exists(file));
	return file;
}

int main(int argc, char *argv[]) {
	if (argc < 4) {
		std::cout << "Usage: trackingBenchmarkApp directory testconfig algorithmconfig1 [algorithmconfig2 [algorithmconfig3 [...]]]" << std::endl;
		std::cout << "where" << std::endl;
		std::cout << " directory ... directory to write the test results and logs into" << std::endl;
		std::cout << " testconfig ... configuration file of the test sequences to run" << std::endl;
		std::cout << " algorithmconfig# ... configuration file of an algorithm to test" << std::endl;
		return 0;
	}

	// create directory
	path directory(argv[1]);
	if (!exists(directory)) {
		create_directory(directory);
	} else if (is_directory(directory)) {
		std::cout << "directory " + directory.string() + " already exists. use this directory? (y/n): ";
		char c;
		std::cin >> c;
		if (std::tolower(c) != 'y')
			return 0;
	} else {
		throw invalid_argument("a file named " + directory.string() + " prevents creating a directory with that name");
	}

	// create app-global logger
	path appLogFile = getNonExistingFile(directory / "log");
	Logger& appLog = Loggers->getLogger("app");
	appLog.addAppender(make_shared<ConsoleAppender>(LogLevel::Info));
	appLog.addAppender(make_shared<FileAppender>(LogLevel::Info, appLogFile.string()));

	// test algorithms
	ptree testConfig, algorithmConfig;
	read_info(argv[2], testConfig);
	size_t runcount = testConfig.get<size_t>("runcount");
	for (int i = 3; i < argc; ++i) {
		read_info(argv[i], algorithmConfig);
		string name = algorithmConfig.get<string>("name");
		path algorithmDirectory = directory / name;
		if (!exists(algorithmDirectory))
			create_directory(algorithmDirectory);
		else if (!is_directory(algorithmDirectory))
			throw invalid_argument("a file named " + algorithmDirectory.string() + " prevents creating a directory with that name");

		path algorithmLogFile = getNonExistingFile(algorithmDirectory / "log");
		Logger& algorithmLog = Loggers->getLogger(name);
		algorithmLog.addAppender(make_shared<FileAppender>(LogLevel::Info, algorithmLogFile.string()));

		algorithmLog.info("Starting test runs for " + name);
		TrackingBenchmark benchmark(algorithmConfig.get_child("tracking"));
		auto iterators = testConfig.equal_range("test");
		double scoreSum = 0;
		size_t testCount = 0;
		size_t exceptionCount = 0;
		for (auto it = iterators.first; it != iterators.second; ++it) {
			try {
				scoreSum += benchmark.runTests(algorithmDirectory, runcount, it->second, algorithmLog);
				testCount++;
			} catch (std::exception& exc) {
				algorithmLog.error(string("A wild exception appeared: ") + exc.what());
				exceptionCount++;
			}
		}
		appLog.info(name + ": " + std::to_string(scoreSum / testCount) + "%" + (exceptionCount > 0 ? " (" + std::to_string(exceptionCount) + " exceptions)" : ""));
	}
	return 0;
}
