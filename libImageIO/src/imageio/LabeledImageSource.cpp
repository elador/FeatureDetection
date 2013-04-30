/*
 * LabeledImageSource.cpp
 *
 *  Created on: 27.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/LandmarkCollection.hpp"
#include "imageio/LandmarkSource.hpp"
#include "imageio/LabeledImageSource.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdexcept>

using cv::imread;
using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;
using std::copy;
using std::sort;
using std::runtime_error;

namespace imageio {

LabeledImageSource::LabeledImageSource(shared_ptr<ImageSource> imageSource, shared_ptr<LandmarkSource> landmarkSource) : lastRetrievedImage(), imageSource(imageSource), landmarkSource(landmarkSource)
{
	 //: files(), index(0) ?
}

LabeledImageSource::~LabeledImageSource() {}

const Mat LabeledImageSource::get() {
	// save the path of current img
	return imageSource->get();
}

const LandmarkCollection LabeledImageSource::getLandmarks() {
	return landmarkSource->get(lastRetrievedImage);
}

const path LabeledImageSource::getPathOfNextImage()
{
	return path();
}

// retrieve the label, if it has one

} /* namespace imageio */
