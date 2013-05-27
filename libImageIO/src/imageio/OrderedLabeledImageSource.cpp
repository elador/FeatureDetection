/*
 * OrderedLabeledImageSource.cpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#include "imageio/OrderedLabeledImageSource.hpp"
#include "imageio/OrderedLandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"

namespace imageio {

OrderedLabeledImageSource::OrderedLabeledImageSource(
		shared_ptr<ImageSource> imageSource, shared_ptr<OrderedLandmarkSource> landmarkSource) :
				imageSource(imageSource), landmarkSource(landmarkSource) {}

OrderedLabeledImageSource::~OrderedLabeledImageSource() {}

const bool OrderedLabeledImageSource::next() {
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

const LandmarkCollection& OrderedLabeledImageSource::getLandmarks() const {
	return landmarkSource->getLandmarks();
}

} /* namespace imageio */
