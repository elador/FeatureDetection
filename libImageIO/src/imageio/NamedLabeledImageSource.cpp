/*
 * NamedLabeledImageSource.cpp
 *
 *  Created on: 27.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/NamedLabeledImageSource.hpp"
#include "imageio/NamedLandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"

using boost::filesystem::path;
using cv::Mat;
using std::shared_ptr;
using std::vector;

namespace imageio {

NamedLabeledImageSource::NamedLabeledImageSource(shared_ptr<ImageSource> imageSource, shared_ptr<NamedLandmarkSource> landmarkSource) :
		LabeledImageSource(imageSource->getSourceName()), imageSource(imageSource), landmarkSource(landmarkSource)
{
}

NamedLabeledImageSource::~NamedLabeledImageSource() {}

void NamedLabeledImageSource::reset()
{
	imageSource->reset();
}

bool NamedLabeledImageSource::next()
{
	return imageSource->next();
}

const Mat NamedLabeledImageSource::getImage() const
{
	return imageSource->getImage();
}

path NamedLabeledImageSource::getName() const
{
	return imageSource->getName();
}

vector<path> NamedLabeledImageSource::getNames() const
{
	return imageSource->getNames();
}

const LandmarkCollection NamedLabeledImageSource::getLandmarks() const {
	return landmarkSource->get(imageSource->getName());
}

} /* namespace imageio */
