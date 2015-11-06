/*
 * DirectPyramidFeatureExtractor.cpp
 *
 *  Created on: 22.03.2013
 *      Author: poschmann
 */

#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/ChainedFilter.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Size;
using cv::Rect;
using cv::Point;
using std::pair;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::invalid_argument;

namespace imageprocessing {

shared_ptr<ImagePyramid> DirectPyramidFeatureExtractor::createPyramid(int width, int minWidth, int maxWidth, int octaveLayerCount) {
	double incrementalScaleFactor = pow(0.5, 1. / octaveLayerCount);
	double minScaleFactor = static_cast<double>(width) / maxWidth;
	double maxScaleFactor = static_cast<double>(width) / minWidth;
	int maxLayerIndex = cvRound(log(minScaleFactor) / log(incrementalScaleFactor));
	int minLayerIndex = cvRound(log(maxScaleFactor) / log(incrementalScaleFactor));
	maxScaleFactor = pow(incrementalScaleFactor, minLayerIndex);
	minScaleFactor = pow(incrementalScaleFactor, maxLayerIndex);
	return make_shared<ImagePyramid>(static_cast<size_t>(octaveLayerCount), minScaleFactor, maxScaleFactor);
}

DirectPyramidFeatureExtractor::DirectPyramidFeatureExtractor(shared_ptr<ImagePyramid> pyramid, int width, int height) :
		pyramid(pyramid), patchWidth(width), patchHeight(height), patchFilter(make_shared<ChainedFilter>()) {}

DirectPyramidFeatureExtractor::DirectPyramidFeatureExtractor(int width, int height, int minWidth, int maxWidth, int octaveLayerCount) :
		pyramid(createPyramid(width, minWidth, maxWidth, octaveLayerCount)),
		patchWidth(width), patchHeight(height), patchFilter(make_shared<ChainedFilter>()) {}

DirectPyramidFeatureExtractor::DirectPyramidFeatureExtractor(int width, int height, int minWidth, int maxWidth, double incrementalScaleFactor) :
		pyramid(createPyramid(width, minWidth, maxWidth, static_cast<size_t>(std::round(log(0.5) / log(incrementalScaleFactor))))),
		patchWidth(width), patchHeight(height), patchFilter(make_shared<ChainedFilter>()) {}

DirectPyramidFeatureExtractor::~DirectPyramidFeatureExtractor() {}

void DirectPyramidFeatureExtractor::addImageFilter(shared_ptr<ImageFilter> filter) {
	pyramid->addImageFilter(filter);
}

void DirectPyramidFeatureExtractor::addLayerFilter(shared_ptr<ImageFilter> filter) {
	pyramid->addLayerFilter(filter);
}

void DirectPyramidFeatureExtractor::addPatchFilter(shared_ptr<ImageFilter> filter) {
	patchFilter->add(filter);
}

void DirectPyramidFeatureExtractor::update(shared_ptr<VersionedImage> image) {
	pyramid->update(image);
}

shared_ptr<Patch> DirectPyramidFeatureExtractor::extract(int x, int y, int width, int height) const {
	const shared_ptr<ImagePyramidLayer> layer = getLayer(width);
	if (!layer)
		return shared_ptr<Patch>();
	Rect patchBounds(getScaled(*layer, x - width / 2), getScaled(*layer, y - height / 2), patchWidth, patchHeight);
	return extract(*layer, patchBounds);
}

vector<shared_ptr<Patch>> DirectPyramidFeatureExtractor::extract(int stepX, int stepY, Rect roi,
		int firstLayer, int lastLayer, int stepLayer) const {
	if (stepX < 1)
		throw invalid_argument("DirectPyramidFeatureExtractor: stepX has to be greater than zero");
	if (stepY < 1)
		throw invalid_argument("DirectPyramidFeatureExtractor: stepY has to be greater than zero");
	if (stepLayer < 1)
		throw invalid_argument("DirectPyramidFeatureExtractor: stepLayer has to be greater than zero");
	Size imageSize = getImageSize();
	if (roi.x == 0 && roi.y == 0 && roi.width == 0 && roi.height == 0) {
		roi.width = imageSize.width;
		roi.height = imageSize.height;
	} else {
		roi.x = std::max(0, roi.x);
		roi.y = std::max(0, roi.y);
		roi.width = std::min(imageSize.width, roi.width + roi.x) - roi.x;
		roi.height = std::min(imageSize.height, roi.height + roi.y) - roi.y;
	}
	vector<shared_ptr<Patch>> patches;
	const vector<shared_ptr<ImagePyramidLayer>>& layers = pyramid->getLayers();
	if (firstLayer < 0)
		firstLayer = layers.front()->getIndex();
	if (lastLayer < 0)
		lastLayer = layers.back()->getIndex();
	for (auto layIt = layers.begin(); layIt != layers.end(); layIt += stepLayer) {
		const shared_ptr<ImagePyramidLayer>& layer = *layIt;
		if (layer->getIndex() < firstLayer)
			continue;
		if (layer->getIndex() > lastLayer)
			break;

		int originalWidth = getOriginal(*layer, patchWidth);
		int originalHeight = getOriginal(*layer, patchHeight);
		const Mat& image = layer->getScaledImage();

		Point roiBegin(getScaled(*layer, roi.x), getScaled(*layer, roi.y));
		Point roiEnd(getScaled(*layer, roi.x + roi.width), getScaled(*layer, roi.y + roi.height));
		Rect patchBounds(roiBegin.x, roiBegin.y, patchWidth, patchHeight);
		for (patchBounds.y = roiBegin.y; patchBounds.y + patchBounds.height < roiEnd.y; patchBounds.y += stepY) {
			for (patchBounds.x = roiBegin.x; patchBounds.x + patchBounds.width < roiEnd.x; patchBounds.x += stepX) {
				int originalX = getOriginal(*layer, patchBounds.x) + originalWidth / 2;
				int originalY = getOriginal(*layer, patchBounds.y) + originalHeight / 2;
				Mat data = patchFilter->applyTo(Mat(image, patchBounds));
				patches.push_back(make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, data));
			}
		}
	}
	return patches;
}

shared_ptr<Patch> DirectPyramidFeatureExtractor::extract(int layerIndex, int x, int y) const {
	shared_ptr<ImagePyramidLayer> layer = pyramid->getLayer(layerIndex);
	if (!layer)
		return shared_ptr<Patch>();
	Rect patchBounds(x - patchWidth / 2, y - patchHeight / 2, patchWidth, patchHeight);
	return extract(*layer, patchBounds);
}

shared_ptr<Patch> DirectPyramidFeatureExtractor::extract(const ImagePyramidLayer& layer, const Rect bounds) const {
	const Mat& image = layer.getScaledImage();
	if (bounds.x < 0 || bounds.y < 0 || bounds.x + bounds.width > image.cols || bounds.y + bounds.height > image.rows)
		return shared_ptr<Patch>();
	int originalWidth = getOriginal(layer, bounds.width);
	int originalHeight = getOriginal(layer, bounds.height);
	int originalX = getOriginal(layer, bounds.x) + originalWidth / 2;
	int originalY = getOriginal(layer, bounds.y) + originalHeight / 2;
	Mat data = patchFilter->applyTo(Mat(image, bounds));
	return make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, data);
}

int DirectPyramidFeatureExtractor::getLayerIndex(int width, int height) const {
	const shared_ptr<ImagePyramidLayer> layer = getLayer(width);
	return layer ? layer->getIndex() : -1;
}

const shared_ptr<ImagePyramidLayer> DirectPyramidFeatureExtractor::getLayer(int width) const {
	double scaleFactor = static_cast<double>(patchWidth) / static_cast<double>(width); // TODO patch width in pixels
	return pyramid->getLayer(scaleFactor);
}

int DirectPyramidFeatureExtractor::getScaled(const ImagePyramidLayer& layer, int value) const {
	return layer.getScaled(value);
}

int DirectPyramidFeatureExtractor::getOriginal(const ImagePyramidLayer& layer, int value) const {
	return layer.getOriginal(value);
}

double DirectPyramidFeatureExtractor::getMinScaleFactor() const {
	return pyramid->getMinScaleFactor();
}

double DirectPyramidFeatureExtractor::getMaxScaleFactor() const {
	return pyramid->getMaxScaleFactor();
}

double DirectPyramidFeatureExtractor::getIncrementalScaleFactor() const {
	return pyramid->getIncrementalScaleFactor();
}

Size DirectPyramidFeatureExtractor::getPatchSize() const {
	return Size(getPatchWidth(), getPatchHeight());
}

Size DirectPyramidFeatureExtractor::getImageSize() const {
	return pyramid->getImageSize();
}

vector<pair<int, double>> DirectPyramidFeatureExtractor::getLayerScales() const {
	return pyramid->getLayerScales();
}

vector<Size> DirectPyramidFeatureExtractor::getLayerSizes() const {
	return pyramid->getLayerSizes();
}

vector<Size> DirectPyramidFeatureExtractor::getPatchSizes() const {
	const vector<shared_ptr<ImagePyramidLayer>>& layers = pyramid->getLayers();
	vector<Size> sizes;
	sizes.reserve(layers.size());
	for (const auto& layer : layers)
		sizes.push_back(Size(getOriginal(*layer, patchWidth), getOriginal(*layer, patchHeight)));
	return sizes;
}

shared_ptr<ImagePyramid> DirectPyramidFeatureExtractor::getPyramid() {
	return pyramid;
}

const shared_ptr<ImagePyramid> DirectPyramidFeatureExtractor::getPyramid() const {
	return pyramid;
}

int DirectPyramidFeatureExtractor::getPatchWidth() const {
	return patchWidth;
}

void DirectPyramidFeatureExtractor::setPatchWidth(int width) {
	patchWidth = width;
}

int DirectPyramidFeatureExtractor::getPatchHeight() const {
	return patchHeight;
}

void DirectPyramidFeatureExtractor::setPatchHeight(int height) {
	patchHeight = height;
}

void DirectPyramidFeatureExtractor::setPatchSize(int width, int height) {
	patchWidth = width;
	patchHeight = height;
}

} /* namespace imageprocessing */
