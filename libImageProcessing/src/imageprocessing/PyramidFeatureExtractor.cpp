/*
 * PyramidFeatureExtractor.cpp
 *
 *  Created on: 22.03.2013
 *      Author: poschmann
 */

#include "imageprocessing/PyramidFeatureExtractor.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/ChainedFilter.hpp"
#include "imageprocessing/Patch.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Rect;
using std::shared_ptr;
using std::make_shared;

namespace imageprocessing {

shared_ptr<ImagePyramid> PyramidFeatureExtractor::createPyramid(int width, int minWidth, int maxWidth, int octaveLayerCount) {
	double incrementalScaleFactor = pow(0.5, 1. / octaveLayerCount);
	double minScaleFactor = static_cast<double>(width) / maxWidth;
	double maxScaleFactor = static_cast<double>(width) / minWidth;
	int maxLayerIndex = cvRound(log(minScaleFactor) / log(incrementalScaleFactor));
	int minLayerIndex = cvRound(log(maxScaleFactor) / log(incrementalScaleFactor));
	maxScaleFactor = pow(incrementalScaleFactor, minLayerIndex);
	minScaleFactor = pow(incrementalScaleFactor, maxLayerIndex);
	return make_shared<ImagePyramid>(static_cast<size_t>(octaveLayerCount), minScaleFactor, maxScaleFactor);
}

PyramidFeatureExtractor::PyramidFeatureExtractor(shared_ptr<ImagePyramid> pyramid, int width, int height) :
		pyramid(pyramid), patchFilter(make_shared<ChainedFilter>()), patchWidth(width), patchHeight(height) {}

PyramidFeatureExtractor::PyramidFeatureExtractor(int width, int height, int minWidth, int maxWidth, int octaveLayerCount) :
		pyramid(createPyramid(width, minWidth, maxWidth, octaveLayerCount)),
		patchFilter(make_shared<ChainedFilter>()), patchWidth(width), patchHeight(height) {}

void PyramidFeatureExtractor::addImageFilter(shared_ptr<ImageFilter> filter) {
	pyramid->addImageFilter(filter);
}

void PyramidFeatureExtractor::addLayerFilter(shared_ptr<ImageFilter> filter) {
	pyramid->addLayerFilter(filter);
}

void PyramidFeatureExtractor::addPatchFilter(shared_ptr<ImageFilter> filter) {
	patchFilter->add(filter);
}

void PyramidFeatureExtractor::update(const Mat& image) {
	pyramid->update(image);
}

void PyramidFeatureExtractor::update(shared_ptr<VersionedImage> image) {
	pyramid->update(image);
}

shared_ptr<Patch> PyramidFeatureExtractor::extract(int x, int y, int width, int height) const {
	const shared_ptr<ImagePyramidLayer> layer = getLayer(width);
	if (!layer)
		return shared_ptr<Patch>();
	Rect bounds(getScaled(*layer, x - width / 2), getScaled(*layer, y - height / 2), patchWidth, patchHeight);
	const Mat& image = layer->getScaledImage();
	if (bounds.x < 0 || bounds.y < 0 || bounds.x + bounds.width > image.cols || bounds.y + bounds.height > image.rows)
		return shared_ptr<Patch>();
	int originalWidth = getOriginal(*layer, bounds.width);
	int originalHeight = getOriginal(*layer, bounds.height);
	int originalX = getOriginal(*layer, bounds.x) + originalWidth / 2;
	int originalY = getOriginal(*layer, bounds.y) + originalHeight / 2;
	Mat data = patchFilter->applyTo(Mat(image, bounds));
	return make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, data);
}

const shared_ptr<ImagePyramidLayer> PyramidFeatureExtractor::getLayer(int width) const {
	double scaleFactor = static_cast<double>(patchWidth) / static_cast<double>(width);
	return pyramid->getLayer(scaleFactor);
}

int PyramidFeatureExtractor::getScaled(const ImagePyramidLayer& layer, int value) const {
	return layer.getScaled(value);
}

int PyramidFeatureExtractor::getOriginal(const ImagePyramidLayer& layer, int value) const {
	return layer.getOriginal(value);
}

const shared_ptr<ImagePyramid> PyramidFeatureExtractor::getPyramid() const {
	return pyramid;
}

int PyramidFeatureExtractor::getPatchWidth() const {
	return patchWidth;
}

int PyramidFeatureExtractor::getPatchHeight() const {
	return patchHeight;
}

} /* namespace imageprocessing */
