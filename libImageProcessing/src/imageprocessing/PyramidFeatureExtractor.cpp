/*
 * PyramidFeatureExtractor.cpp
 *
 *  Created on: 18.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/PyramidFeatureExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/MultipleImageFilter.hpp"

using cv::Point;
using std::make_shared;

namespace imageprocessing {

PyramidFeatureExtractor::PyramidFeatureExtractor(shared_ptr<ImagePyramid> pyramid, int width, int height) :
		pyramid(pyramid), patchWidth(width), patchHeight(height), patchFilter(make_shared<MultipleImageFilter>()) {}

PyramidFeatureExtractor::~PyramidFeatureExtractor() {}

void PyramidFeatureExtractor::addPatchFilter(shared_ptr<ImageFilter> filter) {
	patchFilter->add(filter);
}

shared_ptr<Patch> PyramidFeatureExtractor::extract(int x, int y, int width, int height) const {
	const shared_ptr<ImagePyramidLayer> layer = getLayer(width, height);
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

vector<shared_ptr<Patch>> PyramidFeatureExtractor::extract(int stepX, int stepY, Rect roi, int firstLayer, int lastLayer) const {
	Rect empty;
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
		Point roiBegin(0, 0);
		Point roiEnd(image.cols, image.rows);
		if (roi != empty) {
			roiBegin.x = layer->getScaled(roi.x);
			roiBegin.y = layer->getScaled(roi.y);
			roiEnd.x = layer->getScaled(roi.x + roi.width);
			roiEnd.y = layer->getScaled(roi.y + roi.height);
		}

		Rect patchBounds(roiBegin.x, roiBegin.y, patchWidth, patchHeight);
		Point center(patchBounds.x + patchWidth / 2, patchBounds.y + patchHeight / 2);
		while (patchBounds.y + patchBounds.height <= roiEnd.y) {
			patchBounds.x = roiBegin.x;
			center.x = patchBounds.x + patchWidth / 2;
			while (patchBounds.x + patchBounds.width <= roiEnd.x) {
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

} /* namespace imageprocessing */
