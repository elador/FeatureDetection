/*
 * DirectPyramidFeatureExtractor.cpp
 *
 *  Created on: 22.03.2013
 *      Author: poschmann
 */

#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/MultipleImageFilter.hpp"
#include "boost/iterator/indirect_iterator.hpp"

using cv::Point;
using boost::make_indirect_iterator;
using std::make_shared;

namespace imageprocessing {

DirectPyramidFeatureExtractor::DirectPyramidFeatureExtractor(shared_ptr<ImagePyramid> pyramid, int width, int height) :
		pyramid(pyramid), patchWidth(width), patchHeight(height), patchFilter(make_shared<MultipleImageFilter>()) {}

DirectPyramidFeatureExtractor::DirectPyramidFeatureExtractor(int width, int height, int minWidth, int maxWidth, double incrementalScaleFactor) :
		pyramid(make_shared<ImagePyramid>(static_cast<double>(width) / maxWidth, static_cast<double>(width) / minWidth, incrementalScaleFactor)),
		patchWidth(width), patchHeight(height), patchFilter(make_shared<MultipleImageFilter>()) {}

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

vector<Size> DirectPyramidFeatureExtractor::getPatchSizes() const {
	const vector<shared_ptr<ImagePyramidLayer>>& layers = pyramid->getLayers();
	vector<Size> sizes;
	sizes.resize(layers.size());
	for (auto layer = make_indirect_iterator(layers.begin()); layer != make_indirect_iterator(layers.end()); ++layer)
		sizes.push_back(Size(layer->getOriginal(patchWidth), layer->getOriginal(patchHeight)));
	return sizes;
}

shared_ptr<Patch> DirectPyramidFeatureExtractor::extract(int x, int y, int width, int height) const {
	const shared_ptr<ImagePyramidLayer> layer = getLayer(width);
	if (!layer)
		return shared_ptr<Patch>();
	const Mat& image = layer->getScaledImage();
	int scaledX = layer->getScaled(x);
	int scaledY = layer->getScaled(y);
	int patchBeginX = scaledX - patchWidth / 2; // inclusive
	int patchBeginY = scaledY - patchHeight / 2; // inclusive
	int patchEndX = patchBeginX + patchWidth; // exclusive
	int patchEndY = patchBeginY + patchHeight; // exclusive
	if (patchBeginX < 0 || patchEndX > image.cols
			|| patchBeginY < 0 || patchEndY > image.rows)
		return shared_ptr<Patch>();
	int originalX = layer->getOriginal(scaledX);
	int originalY = layer->getOriginal(scaledY);
	int originalWidth = layer->getOriginal(patchWidth);
	int originalHeight = layer->getOriginal(patchHeight);
	const Mat data(image, Rect(patchBeginX, patchBeginY, patchWidth, patchHeight));
	return make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, patchFilter->applyTo(data));
}

vector<shared_ptr<Patch>> DirectPyramidFeatureExtractor::extract(int stepX, int stepY, Rect roi, int firstLayer, int lastLayer) const {
	if (roi.x == 0 && roi.y == 0 && roi.width == 0 && roi.height == 0) {
		Size size = getImageSize();
		roi.width = size.width;
		roi.height = size.height;
	}
	vector<shared_ptr<Patch>> patches;
	const vector<shared_ptr<ImagePyramidLayer>>& layers = pyramid->getLayers();
	if (firstLayer < 0)
		firstLayer = layers.front()->getIndex();
	if (lastLayer < 0)
		lastLayer = layers.back()->getIndex();
	for (auto layIt = layers.begin(); layIt != layers.end(); ++layIt) {
		shared_ptr<ImagePyramidLayer> layer = *layIt;
		if (layer->getIndex() < firstLayer)
			continue;
		if (layer->getIndex() > lastLayer)
			break;

		int originalWidth = layer->getOriginal(patchWidth);
		int originalHeight = layer->getOriginal(patchHeight);
		const Mat& image = layer->getScaledImage();

		Point roiBegin(layer->getScaled(roi.x), layer->getScaled(roi.y));
		Point roiEnd(layer->getScaled(roi.x + roi.width), layer->getScaled(roi.y + roi.height));
		Rect centerRoi = getCenterRoi(Rect(roiBegin, roiEnd));
		Point centerRoiBegin = centerRoi.tl();
		Point centerRoiEnd = centerRoi.br();
		Point center(centerRoiBegin.x, centerRoiBegin.y);
		Rect patchBounds(roiBegin.x, roiBegin.y, patchWidth, patchHeight);
		while (center.y <= centerRoiEnd.y) {
			patchBounds.x = roiBegin.x;
			center.x = centerRoiBegin.x;
			while (center.x <= centerRoiEnd.x) {
				Mat data(image, patchBounds);
				int originalX = layer->getOriginal(center.x);
				int originalY = layer->getOriginal(center.y);
				patches.push_back(make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, patchFilter->applyTo(data)));
				patchBounds.x += stepX;
				center.x += stepX;
			}
			patchBounds.y += stepY;
			center.y += stepY;
		}
	}
	return patches;
}

shared_ptr<Patch> DirectPyramidFeatureExtractor::extract(int layerIndex, int x, int y) const {
	shared_ptr<ImagePyramidLayer> layer = pyramid->getLayer(layerIndex);
	if (!layer)
		return shared_ptr<Patch>();
	const Mat& image = layer->getScaledImage();
	Rect patchBounds(x - patchWidth / 2, y - patchHeight / 2, patchWidth, patchHeight);
	if (patchBounds.x < 0 || patchBounds.y < 0
			|| patchBounds.x + patchBounds.width >= image.cols
			|| patchBounds.y + patchBounds.height >= image.rows)
		return shared_ptr<Patch>();
	int originalX = layer->getOriginal(x);
	int originalY = layer->getOriginal(y);
	int originalWidth = layer->getOriginal(patchWidth);
	int originalHeight = layer->getOriginal(patchHeight);
	Mat data(image, patchBounds);
	return make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, patchFilter->applyTo(data));
}

} /* namespace imageprocessing */
