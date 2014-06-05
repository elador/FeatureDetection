/*
 * FaceTracking.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "FaceTracking.hpp"
#include "logging/LoggerFactory.hpp"
#include "logging/Logger.hpp"
#include "logging/ConsoleAppender.hpp"
#include "imageio/CameraImageSource.hpp"
#include "imageio/VideoImageSource.hpp"
#include "imageio/KinectImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/VideoImageSink.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/HistEq64Filter.hpp"
#include "imageprocessing/WhiteningFilter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"
#include "imageprocessing/ZeroMeanUnitVarianceFilter.hpp"
#include "imageprocessing/ReshapingFilter.hpp"
#include "imageprocessing/ConversionFilter.hpp"
#include "imageprocessing/UnitNormFilter.hpp"
#include "classification/WvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/RvmClassifier.hpp"
#include "classification/ProbabilisticWvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/ProbabilisticRvmClassifier.hpp"
#include "condensation/ResamplingSampler.hpp"
#include "condensation/GridSampler.hpp"
#include "condensation/LowVarianceSampling.hpp"
#include "condensation/SimpleTransitionModel.hpp"
#include "condensation/WvmSvmModel.hpp"
#include "condensation/SingleClassifierModel.hpp"
#include "condensation/FilteringStateExtractor.hpp"
#include "condensation/WeightedMeanStateExtractor.hpp"
#include "condensation/Sample.hpp"
#include "boost/optional.hpp"
#include "boost/program_options.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <sstream>

using namespace logging;
using namespace imageprocessing;
using namespace classification;
using namespace std::chrono;
using cv::Rect;
using std::milli;
using std::move;
using std::ostringstream;

namespace po = boost::program_options;

const string FaceTracking::videoWindowName = "Image";
const string FaceTracking::controlWindowName = "Controls";

FaceTracking::FaceTracking(unique_ptr<ImageSource> imageSource, unique_ptr<ImageSink> imageSink) :
				imageSource(move(imageSource)),
				imageSink(move(imageSink)) {
	initTracking();
	initGui();
}

FaceTracking::~FaceTracking() {}

void FaceTracking::initTracking() {
	// create measurement model
	shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(20, 20, 80, 480, 5);
	featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
	featureExtractor->addPatchFilter(make_shared<HistEq64Filter>());
	string svmConfigFile1 = "/home/poschmann/projects/ffd/config/fdetection/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--With-outnew02-HQ64SVM.mat";
//	string svmConfigFile1 = "C:/Users/Patrik/Documents/GitHub/config/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--With-outnew02-HQ64SVM.mat";
	string svmConfigFile2 = "/home/poschmann/projects/ffd/config/fdetection/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--ts107742-hq64_thres_0.005--with-outnew02HQ64SVM.mat";
//	string svmConfigFile2 = "C:/Users/Patrik/Documents/GitHub/config/WRVM/fd_web/fnf-hq64-wvm_big-outnew02-hq64SVM/fd_hq64-fnf_wvm_r0.04_c1_o8x8_n14l20t10_hcthr0.72-0.27,0.36-0.14--ts107742-hq64_thres_0.005--with-outnew02HQ64SVM.mat";
	shared_ptr<ProbabilisticWvmClassifier> wvm = ProbabilisticWvmClassifier::loadFromMatlab(svmConfigFile1, svmConfigFile2);
	shared_ptr<ProbabilisticSvmClassifier> svm = ProbabilisticSvmClassifier::loadFromMatlab(svmConfigFile1, svmConfigFile2);
	measurementModel = make_shared<WvmSvmModel>(featureExtractor, wvm, svm);

//	shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(19, 19, 40, 480, 5);
//	featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
//	featureExtractor->addPatchFilter(make_shared<WhiteningFilter>());
//	featureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
//	featureExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0 / 127.5, -1.0));
//	featureExtractor->addPatchFilter(make_shared<UnitNormFilter>(cv::NORM_L2));
//	string classifierFile = "/home/poschmann/projects/ffd/head1919_rvm_whi/head1919_rvm_r1_c1.mat";
//	string thresholdsFile = "/home/poschmann/projects/ffd/head1919_rvm_whi/head1919_rvm_r1_c1--head1919_ts_thres_0.5.mat";
//	shared_ptr<RvmClassifier> rvm = RvmClassifier::loadFromMatlab(classifierFile, thresholdsFile);
//	pair<double, double> logisticParams = ProbabilisticRvmClassifier::loadSigmoidParamsFromMatlab(thresholdsFile);
//	shared_ptr<ProbabilisticClassifier> probRvm = make_shared<ProbabilisticRvmClassifier>(rvm, logisticParams.first, logisticParams.second);
//	measurementModel = make_shared<SingleClassifierModel>(featureExtractor, probRvm);

//	shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(31, 31, 80, 480, 5);
//	featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
//	featureExtractor->addPatchFilter(make_shared<WhiteningFilter>());
//	featureExtractor->addPatchFilter(make_shared<HistogramEqualizationFilter>());
//	featureExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0 / 127.5, -1.0));
//	featureExtractor->addPatchFilter(make_shared<UnitNormFilter>(cv::NORM_L2));
//	featureExtractor->addPatchFilter(make_shared<ReshapingFilter>(1));
//	string svmConfigFile = "/home/poschmann/projects/ffd/whi-head-svm/discGrayscaleWhi";
//	shared_ptr<ProbabilisticSvmClassifier> svm = make_shared<ProbabilisticSvmClassifier>(SvmClassifier::loadText(svmConfigFile));
//	svm->getSvm()->setThreshold(600);
//	measurementModel = make_shared<SingleClassifierModel>(featureExtractor, svm);

//	shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = make_shared<DirectPyramidFeatureExtractor>(31, 31, 80, 480, 5);
//	featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
//	featureExtractor->addPatchFilter(make_shared<ConversionFilter>(CV_32F, 1.0/255.0));
//	featureExtractor->addPatchFilter(make_shared<ReshapingFilter>(1));
//	string svmConfigFile = "/home/poschmann/projects/ffd/whi-head-svm/discGrayscale";
//	shared_ptr<ProbabilisticSvmClassifier> svm = make_shared<ProbabilisticSvmClassifier>(SvmClassifier::loadText(svmConfigFile));
//	svm->getSvm()->setThreshold(0);
//	measurementModel = make_shared<SingleClassifierModel>(featureExtractor, svm);

	// create tracker
	unsigned int count = 800;
	double randomRate = 0.35;
	transitionModel = make_shared<SimpleTransitionModel>(10.0, 0.1);
	resamplingSampler = make_shared<ResamplingSampler>(count, randomRate, make_shared<LowVarianceSampling>(),
			transitionModel, 80, 480);
	gridSampler = make_shared<GridSampler>(80, 480, 1 / featureExtractor->getPyramid()->getIncrementalScaleFactor(), 0.1);
	tracker = unique_ptr<CondensationTracker>(new CondensationTracker(
			resamplingSampler, measurementModel, make_shared<FilteringStateExtractor>(make_shared<WeightedMeanStateExtractor>())));
}

void FaceTracking::initGui() {
	drawSamples = true;

	cvNamedWindow(videoWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(videoWindowName.c_str(), 50, 50);

	cvNamedWindow(controlWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(controlWindowName.c_str(), 900, 50);

	cv::createTrackbar("Grid/Resampling", controlWindowName, NULL, 1, samplerChanged, this);
	cv::setTrackbarPos("Grid/Resampling", controlWindowName, tracker->getSampler() == gridSampler ? 0 : 1);

	cv::createTrackbar("Sample Count", controlWindowName, NULL, 2000, sampleCountChanged, this);
	cv::setTrackbarPos("Sample Count", controlWindowName, resamplingSampler->getCount());

	cv::createTrackbar("Random Rate", controlWindowName, NULL, 100, randomRateChanged, this);
	cv::setTrackbarPos("Random Rate", controlWindowName, 100 * resamplingSampler->getRandomRate());

	cv::createTrackbar("Position Deviation * 10", controlWindowName, NULL, 100, positionDeviationChanged, this);
	cv::setTrackbarPos("Position Deviation * 10", controlWindowName, 100 * transitionModel->getPositionDeviation());

	cv::createTrackbar("Size Deviation * 100", controlWindowName, NULL, 100, sizeDeviationChanged, this);
	cv::setTrackbarPos("Size Deviation * 100", controlWindowName, 100 * transitionModel->getSizeDeviation());

	cv::createTrackbar("Draw samples", controlWindowName, NULL, 1, drawSamplesChanged, this);
	cv::setTrackbarPos("Draw samples", controlWindowName, drawSamples ? 1 : 0);
}

void FaceTracking::samplerChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	if (state == 0)
		tracking->tracker->setSampler(tracking->gridSampler);
	else
		tracking->tracker->setSampler(tracking->resamplingSampler);
}

void FaceTracking::sampleCountChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->resamplingSampler->setCount(state);
}

void FaceTracking::randomRateChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->resamplingSampler->setRandomRate(0.01 * state);
}

void FaceTracking::positionDeviationChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->transitionModel->setPositionDeviation(0.1 * state);
}

void FaceTracking::sizeDeviationChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->transitionModel->setSizeDeviation(0.01 * state);
}

void FaceTracking::drawSamplesChanged(int state, void* userdata) {
	FaceTracking *tracking = (FaceTracking*)userdata;
	tracking->drawSamples = (state == 1);
}

void FaceTracking::drawDebug(cv::Mat& image) {
	cv::Scalar black(0, 0, 0); // blue, green, red
	cv::Scalar red(0, 0, 255); // blue, green, red
	if (drawSamples) {
		const std::vector<shared_ptr<Sample>> samples = tracker->getSamples();
		for (const shared_ptr<Sample>& sample : samples) {
			const cv::Scalar& color = sample->isTarget() ? cv::Scalar(0, 0, sample->getWeight() * 255) : black;
			cv::circle(image, cv::Point(sample->getX(), sample->getY()), 3, color);
		}
	}
	cv::circle(image, cv::Point(10, 10), 5, red, -1);
}

void FaceTracking::run() {
	Logger& log = Loggers->getLogger("app");
	running = true;
	paused = false;

	cv::Mat frame, image;
	cv::Scalar red(0, 0, 255); // blue, green, red
	cv::Scalar green(0, 255, 0); // blue, green, red

	duration<double> allIterationTime;
	duration<double> allCondensationTime;
	int frames = 0;

	while (running) {
		steady_clock::time_point frameStart = steady_clock::now();

		if (!imageSource->next()) {
			std::cerr << "Could not capture frame - press 'q' to quit program" << std::endl;
			stop();
			while ('q' != (char)cv::waitKey(10));
		} else {
			frames++;
			frame = imageSource->getImage();
			steady_clock::time_point condensationStart = steady_clock::now();
			boost::optional<Rect> face = tracker->process(frame);
			steady_clock::time_point condensationEnd = steady_clock::now();
			frame.copyTo(image);
			drawDebug(image);
			if (face)
				cv::rectangle(image, *face, red);
			imshow(videoWindowName, image);
			if (imageSink.get() != 0)
				imageSink->add(image);
			steady_clock::time_point frameEnd = steady_clock::now();

			milliseconds iterationTime = duration_cast<milliseconds>(frameEnd - frameStart);
			milliseconds condensationTime = duration_cast<milliseconds>(condensationEnd - condensationStart);
			allIterationTime += iterationTime;
			allCondensationTime += condensationTime;
			float iterationFps = frames / allIterationTime.count();
			float condensationFps = frames / allCondensationTime.count();

			ostringstream text;
			text.precision(2);
			text << frames << " frame: " << iterationTime.count() << " ms (" << iterationFps << " fps);"
					<< " condensation: " << condensationTime.count() << " ms (" << condensationFps << " fps);"
					<< " face " << (face ? "found" : "not found");
			log.info(text.str());

			int delay = paused ? 0 : 5;
			char c = (char)cv::waitKey(delay);
			if (c == 'p')
				paused = !paused;
			else if (c == 'q')
				stop();
		}
	}
}

void FaceTracking::stop() {
	running = false;
}

int main(int argc, char *argv[]) {
	int verboseLevelText;
	int deviceId, kinectId;
	string filename, directory;
	bool useCamera = false, useKinect = false, useFile = false, useDirectory = false;
	string outputFile;
	int outputFps = -1;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "Produce help message")
			("verbose-text,v", po::value<int>(&verboseLevelText)->implicit_value(2)->default_value(0,"minimal text output"), "Enable text-verbosity (optionally specify level). TODO: Not implemented yet.")
			("filename,f", po::value< string >(&filename), "A filename of a video to run the tracking")
			("directory,i", po::value< string >(&directory), "Use a directory as input")
			("device,d", po::value<int>(&deviceId)->implicit_value(0), "A camera device ID for use with the OpenCV camera driver")
			("kinect,k", po::value<int>(&kinectId)->implicit_value(0), "Windows only: Use a Kinect as camera. Optionally specify a device ID.")
			("output,o", po::value< string >(&outputFile)->default_value("","none"), "Filename to a video file for storing the image data.")
			("output-fps,r", po::value<int>(&outputFps)->default_value(-1), "The framerate of the output video.")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
			std::cout << "Usage: faceTrackingApp [options]" << std::endl;
			std::cout << desc;
			return 0;
		}
		if (vm.count("filename"))
			useFile = true;
		if (vm.count("directory"))
			useDirectory = true;
		if (vm.count("device"))
			useCamera = true;
		if (vm.count("kinect"))
			useKinect = true;
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return -1;
	}

	int inputsSpecified = 0;
	if (useCamera)
		inputsSpecified++;
	if (useKinect)
		inputsSpecified++;
	if (useFile)
		inputsSpecified++;
	if (useDirectory)
		inputsSpecified++;
	if (inputsSpecified != 1) {
		std::cout << "Usage: Please specify a camera, Kinect, file or directory (and only one of them) to run the program. Use -h for help." << std::endl;
		return -1;
	}

	Loggers->getLogger("app").addAppender(make_shared<ConsoleAppender>(LogLevel::Info));

	unique_ptr<ImageSource> imageSource;
	if (useCamera)
		imageSource.reset(new CameraImageSource(deviceId));
	else if (useKinect)
		imageSource.reset(new KinectImageSource(kinectId));
	else if (useFile)
		imageSource.reset(new VideoImageSource(filename));
	else if (useDirectory)
		imageSource.reset(new DirectoryImageSource(directory));

	unique_ptr<ImageSink> imageSink;
	if (outputFile != "") {
		if (outputFps < 0) {
			std::cout << "Usage: You have to specify the framerate of the output video file by using option -r. Use -h for help." << std::endl;
			return -1;
		}
		imageSink.reset(new VideoImageSink(outputFile, outputFps));
	}

	unique_ptr<FaceTracking> tracker(new FaceTracking(move(imageSource), move(imageSink)));
	tracker->run();
	return 0;
}
