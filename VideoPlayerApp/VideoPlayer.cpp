/*
 * VideoPlayer.cpp
 *
 *  Created on: 20.02.2014
 *      Author: poschmann
 */

#include "VideoPlayer.hpp"
#include "logging/LoggerFactory.hpp"
#include "logging/Logger.hpp"
#include "logging/ConsoleAppender.hpp"
#include "imageio/BobotAnnotationSource.hpp"
#include "imageio/SimpleAnnotationSource.hpp"
#include "imageio/VideoImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/VideoImageSink.hpp"
#include "boost/filesystem.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdexcept>

using namespace logging;
using namespace imageio;
using cv::Mat;
using cv::Rect;
using cv::Scalar;
using boost::filesystem::path;
using std::cout;
using std::endl;
using std::unique_ptr;
using std::shared_ptr;
using std::vector;
using std::string;
using std::make_shared;
using std::invalid_argument;

const string VideoPlayer::videoWindowName = "Image";
const string VideoPlayer::controlWindowName = "Controls";

VideoPlayer::VideoPlayer() : paused(false), frame(), image(), colors(), strokeWidth(2) {
	colors.push_back(Scalar(204, 204, 204));
	colors.push_back(Scalar(255, 153, 51));
	colors.push_back(Scalar(0, 51, 204));
	colors.push_back(Scalar(51, 204, 51));
	colors.push_back(Scalar(51, 51, 51));

	cvNamedWindow(videoWindowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvMoveWindow(videoWindowName.c_str(), 50, 50);

	cvNamedWindow(controlWindowName.c_str(), CV_WINDOW_NORMAL);
	cvMoveWindow(controlWindowName.c_str(), 900, 50);

	cv::createTrackbar("Stroke width", controlWindowName, NULL, 3, strokeWidthChanged, this);
	cv::setTrackbarPos("Stroke width", controlWindowName, strokeWidth);
}

void VideoPlayer::strokeWidthChanged(int state, void* userdata) {
	VideoPlayer *player = (VideoPlayer*)userdata;
	player->strokeWidth = state;
}

void VideoPlayer::drawAnnotation(Mat& image, const Rect& target, const Scalar& color) {
	cv::rectangle(image, target, color, strokeWidth);
}

void VideoPlayer::play(shared_ptr<ImageSource> imageSource,
		vector<shared_ptr<AnnotationSource>> annotationSources,
		shared_ptr<ImageSink> imageSink) {
	while (imageSource->next()) {
		frame = imageSource->getImage();
		frame.copyTo(image);
		for (size_t i = 0; i < annotationSources.size(); ++i) {
			const shared_ptr<AnnotationSource>& annotationSource = annotationSources[i];
			const Scalar& color = colors[std::min(colors.size() - 1, i)];
			if (annotationSource->next())
				drawAnnotation(image, annotationSource->getAnnotation(), color);
		}
		imshow(videoWindowName, image);
		if (imageSink.get() != 0)
			imageSink->add(image);

		int delay = paused ? 0 : 35;
		char c = (char)cv::waitKey(delay);
		if (c == 'p')
			paused = !paused;
		else if (c == 'q')
			break;
	}
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		cout << "usage: ./VideoPlayer video [-s annotations1 [annotations2 [...]]] [-b annotations1 [annotations2 [...]]] [-o output framerate [codec]]" << endl;
		cout << "where" << endl;
		cout << " video ... video file or image directory" << endl;
		cout << " -s ... specifying that the following annotation files have a simple format" << endl;
		cout << " -b ... specifying that the following annotation files have the BoBoT format" << endl;
		cout << " annotations# ... file containing annotation data in the specified format" << endl;
		cout << " -o ... flag for indicating the output specification" << endl;
		cout << " output ... video file for saving the output (video with bounding boxes)" << endl;
		cout << " framerate ... framerate of the resulting video (should be the same as the input video)" << endl;
		cout << " codec ... four digit video codec (MJPG, DIVX, ...)" << endl;
		cout << "example(1): ./VideoPlayer /path/to/image/directory" << endl;
		cout << "example(2): ./VideoPlayer path/to/myvideo.avi -s path/to/annotations1.txt path/to/annotations2.txt -o video-with-annotations.avi 25" << endl;
		return 0;
	}

	path sourceFile(argv[1]);
	if (!exists(sourceFile))
		throw invalid_argument("video file/directory " + sourceFile.string() + " does not exist");
	shared_ptr<ImageSource> imageSource;
	if (is_directory(sourceFile))
		imageSource.reset(new DirectoryImageSource(sourceFile.string()));
	else
		imageSource.reset(new VideoImageSource(sourceFile.string()));

	enum GroundTruthType { UNKNOWN, SIMPLE, BOBOT };
	GroundTruthType type = GroundTruthType::UNKNOWN;
	vector<shared_ptr<AnnotationSource>> annotationSources;
	int i = 2;
	for (; i < argc; ++i) {
		if (argv[i][0] == '-') {
			string option = string(argv[i]);
			if ("-s" == option)
				type = GroundTruthType::SIMPLE;
			else if ("-b" == option)
				type = GroundTruthType::BOBOT;
			else if ("-o" == option)
				break;
			else
				throw invalid_argument("unknown option " + option);
			continue;
		}
		path annotationFile(argv[i]);
		if (!exists(annotationFile))
			throw invalid_argument("annotation file " + annotationFile.string() + " does not exist");
		if (type == GroundTruthType::SIMPLE)
			annotationSources.push_back(make_shared<SimpleAnnotationSource>(annotationFile.string()));
		else if (type == GroundTruthType::BOBOT)
			annotationSources.push_back(make_shared<BobotAnnotationSource>(annotationFile.string(), imageSource));
		else
			throw invalid_argument("no type specified for annotation file " + annotationFile.string() + " (needs option -s or -b before giving annotation files)");
	}

	shared_ptr<ImageSink> imageSink;
	if (i < argc) {
		if ("-o" != string(argv[i]))
			throw invalid_argument("unknown option " + string(argv[i]));
		if (i + 2 >= argc)
			throw invalid_argument("there is no given filename and framerate after -o");
		path outputFile(argv[i + 1]);
		if (exists(outputFile))
			throw invalid_argument("output file \"" + outputFile.string() + "\" does already exist");
		double outputFps = std::stod(argv[i + 2]);
		int codec = CV_FOURCC('M', 'J', 'P', 'G');
		if (i + 3 < argc) {
			string fourcc = string(argv[i + 3]);
			if (fourcc.length() != 4)
				throw invalid_argument("codec must consist of four characters to be valid");
			codec = CV_FOURCC(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
		}
		imageSink = make_shared<VideoImageSink>(outputFile.string(), outputFps, codec);
	}

	Logger& log = Loggers->getLogger("app");
	log.addAppender(make_shared<ConsoleAppender>(LogLevel::Info));
	try {
		unique_ptr<VideoPlayer> player(new VideoPlayer());
		player->play(imageSource, annotationSources, imageSink);
	} catch (std::exception& exc) {
		log.error(string("A wild exception appeared: ") + exc.what());
		throw;
	}
	return 0;
}
