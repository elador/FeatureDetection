/*
 * PyramidPatchExtractor.cpp
 *
 *  Created on: 18.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/PyramidPatchExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"

using cv::Rect;
using cv::Point;
using std::make_shared;

namespace imageprocessing {

PyramidPatchExtractor::PyramidPatchExtractor(shared_ptr<ImagePyramid> pyramid, int width, int height) :
		pyramid(pyramid), patchWidth(width), patchHeight(height) {}

PyramidPatchExtractor::~PyramidPatchExtractor() {}

void PyramidPatchExtractor::update(const Mat& image) {
	pyramid->update(image);
}

shared_ptr<Patch> PyramidPatchExtractor::extract(int x, int y, int width, int height) {
	double scaleFactor = static_cast<double>(width) / static_cast<double>(patchWidth);
	const shared_ptr<ImagePyramidLayer> layer = pyramid->getLayer(scaleFactor);
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
	return make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, data.clone());	// Patrik: Hmm, do we really want to clone the patch-data here? Wouldn't it be better to work with the same data and just the ROI as long as possible?
}

vector<shared_ptr<Patch>> PyramidPatchExtractor::extract(int stepX, int stepY) {
	vector<shared_ptr<Patch>> patches;
	const vector<shared_ptr<ImagePyramidLayer>>& layers = pyramid->getLayers();
	for (vector<shared_ptr<ImagePyramidLayer>>::const_iterator layIt = layers.begin(); layIt != layers.end(); ++layIt) {
		shared_ptr<ImagePyramidLayer> layer = *layIt;
		int originalWidth = layer->getOriginal(patchWidth);
		int originalHeight = layer->getOriginal(patchHeight);
		const Mat& image = layer->getScaledImage();
		Rect patchBounds(0, 0, patchWidth, patchHeight);
		Point center(patchWidth / 2, patchHeight / 2);
		while (patchBounds.y + patchBounds.height <= image.rows) {
			while (patchBounds.x + patchBounds.width <= image.cols) {
				Mat data(image, patchBounds);
				int originalX = layer->getOriginal(center.x);
				int originalY = layer->getOriginal(center.y);
				patches.push_back(make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, data.clone())); // Patrik: Hmm, do we really want to clone the patch-data here? Wouldn't it be better to work with the same data and just the ROI as long as possible?
				patchBounds.x += stepX;
				center.x += stepX;
			}
			patchBounds.y += stepY;
			center.y += stepY;
			patchBounds.x = 0;		// Patrik: @Peter, please check my changes here and on the next line.
			center.x = patchWidth / 2;		// see above
		}
	}
	return patches;
}

} /* namespace imageprocessing */
