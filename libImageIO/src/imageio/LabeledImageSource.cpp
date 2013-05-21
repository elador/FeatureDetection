/*
 * LabeledImageSource.cpp
 *
 *  Created on: 27.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/LandmarkCollection.hpp"
#include "imageio/DefaultLandmarkSource.hpp"
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

LabeledImageSource::LabeledImageSource(shared_ptr<ImageSource> imageSource, shared_ptr<DefaultLandmarkSource> landmarkSource) : imageSource(imageSource), landmarkSource(landmarkSource)
{
}

LabeledImageSource::~LabeledImageSource() {}

const bool LabeledImageSource::next()
{
	return imageSource->next();
}

const Mat LabeledImageSource::get() {
	return imageSource->get();
}

const Mat LabeledImageSource::getImage() const
{
	return imageSource->getImage();
}

const path LabeledImageSource::getName() const
{
	return imageSource->getName();
}

const vector<path> LabeledImageSource::getNames() const
{
	return imageSource->getNames();
}

const LandmarkCollection LabeledImageSource::getLandmarks() {
	return landmarkSource->get(imageSource->getName());
}

} /* namespace imageio */
