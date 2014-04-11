/*
 * OrderedLabeledImageSource.cpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#include "imageio/OrderedLabeledImageSource.hpp"
#include "imageio/LandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"

using boost::filesystem::path;
using cv::Mat;
using std::vector;
using std::shared_ptr;

namespace imageio {

OrderedLabeledImageSource::OrderedLabeledImageSource(
		shared_ptr<ImageSource> imageSource, shared_ptr<LandmarkSource> landmarkSource) :
				LabeledImageSource(imageSource->getSourceName()), imageSource(imageSource), landmarkSource(landmarkSource) {}

OrderedLabeledImageSource::~OrderedLabeledImageSource() {}

void OrderedLabeledImageSource::reset() {
	imageSource->reset();
	landmarkSource->reset();
}

bool OrderedLabeledImageSource::next() {
	landmarkSource->next();
	return imageSource->next();
}

const Mat OrderedLabeledImageSource::getImage() const {
	return imageSource->getImage();
}

path OrderedLabeledImageSource::getName() const {
	return imageSource->getName();
}

vector<path> OrderedLabeledImageSource::getNames() const {
	return imageSource->getNames();
}

const LandmarkCollection OrderedLabeledImageSource::getLandmarks() const {
	return landmarkSource->getLandmarks();
}

} /* namespace imageio */
