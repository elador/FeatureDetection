/*
 * BobotAnnotationSink.cpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#include "imageio/BobotAnnotationSink.hpp"
#include "imageio/ImageSource.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Rect;
using std::string;
using std::shared_ptr;
using std::runtime_error;

namespace imageio {

BobotAnnotationSink::BobotAnnotationSink(const string& videoFilename, int imageWidth, int imageHeight) :
		videoFilename(videoFilename), imageWidth(imageWidth), imageHeight(imageHeight), imageSource(), output(), index(0) {
	output.setf(std::ios_base::fixed, std::ios_base::floatfield);
	output.precision(6);
}

BobotAnnotationSink::BobotAnnotationSink(const string& videoFilename, shared_ptr<ImageSource> imageSource) :
		videoFilename(videoFilename), imageWidth(0), imageHeight(0), imageSource(imageSource), output(), index(0) {
	output.setf(std::ios_base::fixed, std::ios_base::floatfield);
	output.precision(6);
}

bool BobotAnnotationSink::isOpen() {
	return output.is_open();
}

void BobotAnnotationSink::open(const string& filename) {
	if (isOpen())
		throw runtime_error("BobotAnnotationSink: sink is already open");
	output.open(filename);
	output << videoFilename << '\n';
}

void BobotAnnotationSink::close() {
	output.close();
	index = 0;
}

void BobotAnnotationSink::add(const Rect& annotation) {
	if (!isOpen())
		throw runtime_error("BobotAnnotationSink: sink is not open");
	if (imageSource) {
		Mat image = imageSource->getImage();
		imageWidth = image.cols;
		imageHeight = image.rows;
	}
	if (annotation.area() == 0) {
		output << index++ << " 0 0 0 0\n";
	} else {
		float x = static_cast<float>(annotation.x) / imageWidth;
		float y = static_cast<float>(annotation.y) / imageHeight;
		float width = static_cast<float>(annotation.width) / imageWidth;
		float height = static_cast<float>(annotation.height) / imageHeight;
		output << index++ << ' ' << x << ' ' << y << ' ' << width << ' ' << height << '\n';
	}
}

} /* namespace imageio */
