/*
 * NamedLabeledImageSource.cpp
 *
 *  Created on: 27.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/NamedLabeledImageSource.hpp"
#include "imageio/NamedLandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"
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

NamedLabeledImageSource::NamedLabeledImageSource(shared_ptr<ImageSource> imageSource, shared_ptr<NamedLandmarkSource> landmarkSource) : imageSource(imageSource), landmarkSource(landmarkSource)
{
}

NamedLabeledImageSource::~NamedLabeledImageSource() {}

const bool NamedLabeledImageSource::next()
{
	return imageSource->next();
}

const Mat NamedLabeledImageSource::get() {
	return imageSource->get();
}

const Mat NamedLabeledImageSource::getImage() const
{
	return imageSource->getImage();
}

const path NamedLabeledImageSource::getName() const
{
	return imageSource->getName();
}

const vector<path> NamedLabeledImageSource::getNames() const
{
	return imageSource->getNames();
}

const LandmarkCollection NamedLabeledImageSource::getLandmarks() {
	return landmarkSource->get(imageSource->getName());
}

} /* namespace imageio */
