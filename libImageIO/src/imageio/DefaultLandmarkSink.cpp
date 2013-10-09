/*
 * DefaultLandmarkSink.cpp
 *
 *  Created on: 08.10.2013
 *      Author: Patrik Huber
 */

#include "imageio/DefaultLandmarkSink.hpp"
#include "imageio/Landmark.hpp"
#include "imageio/LandmarkCollection.hpp"

#include "opencv2/core/core.hpp"

#include <fstream>

using std::ofstream;

using cv::Rect_;

namespace imageio {

DefaultLandmarkSink::DefaultLandmarkSink(const boost::filesystem::path& outputPath) :
		outputPath(outputPath) {
}


DefaultLandmarkSink::~DefaultLandmarkSink() {}


void DefaultLandmarkSink::write(const LandmarkCollection& collection, const boost::filesystem::path imageFilename) {
	ofstream output(outputPath.string() + imageFilename.stem().string() + ".txt", std::ios_base::out);
	if (!output.is_open()) {
		throw std::runtime_error("DefaultOrderedLandmarkSink: Couldn't open the landmark-file for writing: " + outputPath.string() + imageFilename.stem().string() + ".txt");
	}
	output.precision(4);

	vector<shared_ptr<Landmark>> landmarks = collection.getLandmarks();
	for (const auto& l : landmarks) {
		Rect_<float> rect = l->getRect();
		// Format: 'landmarkName topLeftX topLeftY width height'
		output << l->getName() << ' ' << rect.tl().x << ' ' << rect.tl().y << ' ' << rect.width << ' ' << rect.height << std::endl;
	}

	output.close();
}


} /* namespace imageio */
