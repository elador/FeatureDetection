/*
 * PatchExtractor.cpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#include "imageprocessing/PatchExtractor.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/PyramidLayer.hpp"

using cv::Rect;

namespace imageprocessing {

PatchExtractor::PatchExtractor(shared_ptr<ImagePyramid> pyramid, int width, int height) :
		pyramid(pyramid), patchWidth(width), patchHeight(height) {}

PatchExtractor::~PatchExtractor() {}

void PatchExtractor::update(const Mat& image) {
	pyramid->update(image);
}

unique_ptr<Patch> PatchExtractor::extract(int x, int y, int width, int height) {
	PyramidLayer* level = pyramid->getLayer(static_cast<double>(width) / static_cast<double>(patchWidth));
	if (level == NULL)
		return unique_ptr<Patch>();
	const Mat& image = level->getScaledImage();
	int scaledX = level->getScaled(x);
	int scaledY = level->getScaled(y);
	int patchBeginX = scaledX - patchWidth / 2; // inclusive
	int patchBeginY = scaledY - patchHeight / 2; // inclusive
	int patchEndX = patchBeginX + patchWidth; // exclusive
	int patchEndY = patchBeginY + patchHeight; // exclusive
	if (patchBeginX < 0 || patchEndX > image.cols
			|| patchBeginY < 0 || patchEndY > image.rows)
		return unique_ptr<Patch>();
	Mat data(image, Rect(patchBeginX, patchBeginY, patchWidth, patchHeight));
	return unique_ptr<Patch>(new Patch(scaledX, scaledY, level->getScaleFactor(), data));
}

vector<Patch> PatchExtractor::extract(int stepX, int stepY) {
	// TODO ausprogrammieren
	return vector<Patch>;
}

PyramidLayer* PatchExtractor::getPyramidLayer(int size) {
	double factor = (double)size / (double)featureSize.width;
	return pyramid->getLayer(factor);
}

} /* namespace imageprocessing */
