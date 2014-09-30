/*
 * SimpleAnnotationSink.cpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#include "imageio/SimpleAnnotationSink.hpp"
#include <stdexcept>

using cv::Rect;
using std::string;
using std::runtime_error;

namespace imageio {

SimpleAnnotationSink::SimpleAnnotationSink() : output() {}

bool SimpleAnnotationSink::isOpen() {
	return output.is_open();
}

void SimpleAnnotationSink::open(const string& filename) {
	if (isOpen())
		throw runtime_error("SimpleAnnotationSink: sink is already open");
	output.open(filename);
}

void SimpleAnnotationSink::close() {
	output.close();
}

void SimpleAnnotationSink::add(const Rect& annotation) {
	if (!isOpen())
		throw runtime_error("SimpleAnnotationSink: sink is not open");
	if (annotation.area() == 0)
		output << "0 0 0 0\n";
	else
		output << annotation.x << ' ' << annotation.y << ' ' << annotation.width << ' ' << annotation.height << '\n';
}

} /* namespace imageio */
