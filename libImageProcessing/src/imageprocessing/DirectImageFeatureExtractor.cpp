/*
 * DirectImageFeatureExtractor.cpp
 *
 *  Created on: 30.06.2013
 *      Author: poschmann
 */

#include "imageprocessing/DirectImageFeatureExtractor.hpp"
#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/ChainedFilter.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/Patch.hpp"

using cv::Mat;
using cv::Rect;
using std::shared_ptr;
using std::make_shared;

namespace imageprocessing {

DirectImageFeatureExtractor::DirectImageFeatureExtractor() :
		version(),
		image(),
		imageFilter(make_shared<ChainedFilter>()),
		patchFilter(make_shared<ChainedFilter>()) {}

void DirectImageFeatureExtractor::addImageFilter(shared_ptr<ImageFilter> filter) {
	imageFilter->add(filter);
}

void DirectImageFeatureExtractor::addPatchFilter(shared_ptr<ImageFilter> filter) {
	patchFilter->add(filter);
}

void DirectImageFeatureExtractor::update(shared_ptr<VersionedImage> image) {
	if (version != image->getVersion()) {
		imageFilter->applyTo(image->getData(), this->image);
		version = image->getVersion();
	}
}

shared_ptr<Patch> DirectImageFeatureExtractor::extract(int x, int y, int width, int height) const {
	int patchBeginX = x - width / 2; // inclusive
	int patchBeginY = y - height / 2; // inclusive
	int patchEndX = patchBeginX + width; // exclusive
	int patchEndY = patchBeginY + height; // exclusive
	if (patchBeginX < 0 || patchEndX > image.cols
			|| patchBeginY < 0 || patchEndY > image.rows)
		return shared_ptr<Patch>();
	const Mat data(image, Rect(patchBeginX, patchBeginY, width, height));
	return make_shared<Patch>(x, y, width, height, patchFilter->applyTo(data));
}

} /* namespace imageprocessing */
