/*
 * RectLandmarkSink.cpp
 *
 *  Created on: 08.10.2013
 *      Author: Patrik Huber
 */

#include "imageio/RectLandmarkSink.hpp"
#include "imageio/Landmark.hpp"
#include "imageio/LandmarkCollection.hpp"

#include "opencv2/core/core.hpp"

#include <fstream>

using cv::Rect_;
using boost::filesystem::path;
using std::ofstream;
using std::vector;
using std::shared_ptr;

namespace imageio {

RectLandmarkSink::RectLandmarkSink(const boost::filesystem::path& outputDirectory) : outputDirectory(outputDirectory)
{
}

void RectLandmarkSink::write(const LandmarkCollection& collection, const path imageFilename)
{
	ofstream output((outputDirectory / imageFilename.stem()).string() + ".txt", std::ios_base::out);
	if (!output.is_open()) {
		throw std::runtime_error("DefaultOrderedLandmarkSink TODO: Couldn't open the landmark-file for writing: " + outputDirectory.string() + imageFilename.stem().string() + ".txt");
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
