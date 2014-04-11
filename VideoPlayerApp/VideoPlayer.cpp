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
#include "imageio/OrderedLandmarkSource.hpp"
#include "imageio/BobotLandmarkSource.hpp"
#include "imageio/SimpleLandmarkSource.hpp"
#include "imageio/VideoImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/VideoImageSink.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageio/Landmark.hpp"
#include "boost/filesystem.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdexcept>

using namespace logging;
using namespace imageio;
using cv::Mat;
using boost::filesystem::path;
using std::cout;
using std::endl;
using std::unique_ptr;
using std::shared_ptr;
using std::vector;
using std::string;
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

void VideoPlayer::drawLandmarks(Mat& image, const LandmarkCollection& landmarks, const Scalar color) {
	for (const shared_ptr<Landmark>& landmark : landmarks.getLandmarks())
		landmark->draw(image, color, strokeWidth);
}

void VideoPlayer::play(shared_ptr<ImageSource> imageSource,
		vector<shared_ptr<OrderedLandmarkSource>> landmarkSources,
		shared_ptr<ImageSink> imageSink) {
	while (imageSource->next()) {
		frame = imageSource->getImage();
		frame.copyTo(image);
		for (size_t i = 0; i < landmarkSources.size(); ++i) {
			const shared_ptr<OrderedLandmarkSource>& landmarkSource = landmarkSources[i];
			const Scalar& color = colors[std::min(colors.size() - 1, i)];
			if (landmarkSource->next())
				drawLandmarks(image, landmarkSource->getLandmarks(), color);
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
		cout << "usage: ./VideoPlayer video [landmarks1 [landmarks2 [landmarks3 ...]]] [-o output framerate]" << endl;
		cout << "example(1): ./VideoPlayer /path/to/image/directory" << endl;
		cout << "example(2): ./VideoPlayer path/to/myvideo.avi path/to/landmarks1.txt -o video-with-landmarks.avi 25" << endl;
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

	vector<shared_ptr<OrderedLandmarkSource>> landmarkSources;
	int i = 2;
	for (; i < argc; ++i) {
		if (argv[i][0] == '-')
			break;
		path landmarkFile(argv[i]);
		if (!exists(landmarkFile))
			throw invalid_argument("landmark file " + landmarkFile.string() + " does not exist");
		landmarkSources.push_back(make_shared<BobotLandmarkSource>(landmarkFile.string(), imageSource));
//		landmarks.push_back(make_shared<SimpleLandmarkSource>(landmarkFile.string())); // TODO
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
		imageSink = make_shared<VideoImageSink>(outputFile.string(), outputFps, CV_FOURCC('D','I','V','X'));
	}

	Logger& log = Loggers->getLogger("app");
	log.addAppender(make_shared<ConsoleAppender>(loglevel::INFO));
	try {
		unique_ptr<VideoPlayer> player(new VideoPlayer());
		player->play(imageSource, landmarkSources, imageSink);
	} catch (std::exception& exc) {
		log.error(string("A wild exception appeared: ") + exc.what());
		throw;
	}
	return 0;
}
