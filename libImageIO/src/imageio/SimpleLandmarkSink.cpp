/*
 * SingleLandmarkSink.cpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#include "imageio/SingleLandmarkSink.hpp"
#include "imageio/Landmark.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "opencv2/core/core.hpp"
#include <stdexcept>

using cv::Rect_;
using std::string;
using std::shared_ptr;
using std::ofstream;
using std::runtime_error;

namespace imageio {

SingleLandmarkSink::SingleLandmarkSink(size_t precision, const string& landmarkName) : landmarkName(landmarkName), output() {
	output.setf(std::ios_base::fixed, std::ios_base::floatfield);
	output.precision(precision);
}

SingleLandmarkSink::SingleLandmarkSink(const string& filename, size_t precision, const string& landmarkName) :
		landmarkName(landmarkName), output(filename) {
	output.setf(std::ios_base::fixed, std::ios_base::floatfield);
	output.precision(precision);
}

bool SingleLandmarkSink::isOpen() {
	return output.is_open();
}

void SingleLandmarkSink::open(const string& filename) {
	if (isOpen())
		throw runtime_error("SingleLandmarkSink: sink is already open");
	output.open(filename);
}

void SingleLandmarkSink::close() {
	output.close();
}

void SingleLandmarkSink::add(const LandmarkCollection& collection) {
	if (!isOpen())
		throw runtime_error("SingleLandmarkSink: sink is not open");
	const shared_ptr<Landmark> landmark = getLandmark(collection);
	if (landmark->isVisible()) {
		Rect_<float> rect = landmark->getRect();
		output << rect.x << ' ' << rect.y << ' ' << rect.width << ' ' << rect.height << '\n';
	} else {
		output << "0 0 0 0\n";
	}
}

const shared_ptr<Landmark> SingleLandmarkSink::getLandmark(const LandmarkCollection& collection) {
	if (landmarkName.empty())
		return collection.getLandmark();
	return collection.getLandmark(landmarkName);
}

} /* namespace imageio */
