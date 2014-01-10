/*
 * CellBasedPyramidFeatureExtractor.cpp
 *
 *  Created on: 16.12.2013
 *      Author: poschmann
 */

#include "imageprocessing/CellBasedPyramidFeatureExtractor.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/ChainedFilter.hpp"
#include "imageprocessing/Patch.hpp"
#include <stdexcept>

using cv::Rect;
using cv::Point;
using std::make_shared;
using std::invalid_argument;

namespace imageprocessing {

static shared_ptr<ImagePyramid> createPyramid(
		size_t cellSize, size_t cols, size_t minWidth, size_t maxWidth, double incrementalScaleFactor) {
	double width = cellSize * cols;
	double minScaleFactor = width / maxWidth;
	double maxScaleFactor = width / minWidth;
	return make_shared<ImagePyramid>(minScaleFactor, maxScaleFactor, incrementalScaleFactor);
}

CellBasedPyramidFeatureExtractor::CellBasedPyramidFeatureExtractor(
		size_t cellSize, size_t cols, size_t rows, size_t minWidth, size_t maxWidth, double incrementalScaleFactor) :
				pyramid(createPyramid(cellSize, cols, minWidth, maxWidth, incrementalScaleFactor)),
				cellSize(cellSize), cellColumnCount(cols), cellRowCount(rows), patchFilter(make_shared<ChainedFilter>()) {
	if (cellSize == 0)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: the cell size must be greater than zero");
	if (cols == 0 || rows == 0)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: the amount of columns and rows must be greater than zero");
}

void CellBasedPyramidFeatureExtractor::addImageFilter(shared_ptr<ImageFilter> filter) {
	pyramid->addImageFilter(filter);
}

void CellBasedPyramidFeatureExtractor::addLayerFilter(shared_ptr<ImageFilter> filter) {
	pyramid->addLayerFilter(filter);
}

void CellBasedPyramidFeatureExtractor::addPatchFilter(shared_ptr<ImageFilter> filter) {
	patchFilter->add(filter);
}

void CellBasedPyramidFeatureExtractor::update(const Mat& image) {
	pyramid->update(image);
}

void CellBasedPyramidFeatureExtractor::update(shared_ptr<VersionedImage> image) {
	pyramid->update(image);
}

shared_ptr<Patch> CellBasedPyramidFeatureExtractor::extract(int x, int y, int width, int height) const {
	const shared_ptr<ImagePyramidLayer> layer = getLayer(width);
	if (!layer)
		return shared_ptr<Patch>();
	const Mat& image = layer->getScaledImage();
	int scaledX = cvRound(x * layer->getScaleFactor() / cellSize);
	int scaledY = cvRound(y * layer->getScaleFactor() / cellSize);
	int patchBeginX = scaledX - cellColumnCount / 2; // inclusive
	int patchBeginY = scaledY - cellRowCount / 2; // inclusive
	int patchEndX = patchBeginX + cellColumnCount; // exclusive
	int patchEndY = patchBeginY + cellRowCount; // exclusive
	if (patchBeginX < 0 || patchEndX > image.cols
			|| patchBeginY < 0 || patchEndY > image.rows)
		return shared_ptr<Patch>();
	int originalX = cvRound(scaledX * cellSize / layer->getScaleFactor());
	int originalY = cvRound(scaledY * cellSize / layer->getScaleFactor());
	int originalWidth = layer->getOriginal(cellSize * cellColumnCount);
	int originalHeight = layer->getOriginal(cellSize * cellRowCount);
	const Mat data(image, Rect(patchBeginX, patchBeginY, cellColumnCount, cellRowCount));
	return make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, patchFilter->applyTo(data));
}

vector<shared_ptr<Patch>> CellBasedPyramidFeatureExtractor::extract(int stepX, int stepY, Rect roi,
		int firstLayer, int lastLayer, int stepLayer) const {
	if (stepX < 1)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: stepX has to be greater than zero");
	if (stepY < 1)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: stepY has to be greater than zero");
	if (stepLayer < 1)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: stepLayer has to be greater than zero");
	stepX = std::max(1, stepX / static_cast<int>(cellSize));
	stepY = std::max(1, stepY / static_cast<int>(cellSize));
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
		shared_ptr<ImagePyramidLayer> layer = *layIt;
		if (layer->getIndex() < firstLayer)
			continue;
		if (layer->getIndex() > lastLayer)
			break;

		int originalWidth = layer->getOriginal(cellSize * cellColumnCount);
		int originalHeight = layer->getOriginal(cellSize * cellRowCount);
		const Mat& image = layer->getScaledImage();
		double scaleFactor = layer->getScaleFactor();

		Point roiBegin(cvRound(roi.x * scaleFactor / cellSize), cvRound(roi.y * scaleFactor / cellSize));
		Point roiEnd(cvRound((roi.x + roi.width) * scaleFactor / cellSize), cvRound((roi.y + roi.height) * scaleFactor / cellSize));
		Rect centerRoi = getCenterRoi(Rect(roiBegin, roiEnd));
		Point centerRoiBegin = centerRoi.tl();
		Point centerRoiEnd = centerRoi.br();
		Point center(centerRoiBegin.x, centerRoiBegin.y);
		Rect patchBounds(roiBegin.x, roiBegin.y, cellColumnCount, cellRowCount);
		while (center.y <= centerRoiEnd.y) {
			patchBounds.x = roiBegin.x;
			center.x = centerRoiBegin.x;
			while (center.x <= centerRoiEnd.x) {
				Mat data(image, patchBounds);
				int originalX = cvRound(center.x * cellSize / scaleFactor);
				int originalY = cvRound(center.y * cellSize / scaleFactor);
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

shared_ptr<Patch> CellBasedPyramidFeatureExtractor::extract(int layerIndex, int x, int y) const {
	shared_ptr<ImagePyramidLayer> layer = pyramid->getLayer(layerIndex);
	if (!layer)
		return shared_ptr<Patch>();
	const Mat& image = layer->getScaledImage();
	Rect patchBounds(x - cellColumnCount / 2, y - cellRowCount / 2, cellColumnCount, cellRowCount);
	if (patchBounds.x < 0 || patchBounds.y < 0
			|| patchBounds.x + patchBounds.width >= image.cols
			|| patchBounds.y + patchBounds.height >= image.rows)
		return shared_ptr<Patch>();
	int originalX = cvRound(x * cellSize / layer->getScaleFactor());
	int originalY = cvRound(y * cellSize / layer->getScaleFactor());
	int originalWidth = layer->getOriginal(cellSize * cellColumnCount);
	int originalHeight = layer->getOriginal(cellSize * cellRowCount);
	Mat data(image, patchBounds);
	return make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, patchFilter->applyTo(data));
}

Rect CellBasedPyramidFeatureExtractor::getCenterRoi(const Rect& roi) const {
	return Rect(roi.x + cellColumnCount / 2, roi.y + cellRowCount / 2, roi.width - cellColumnCount, roi.height - cellRowCount);
}

int CellBasedPyramidFeatureExtractor::getLayerIndex(int width, int height) const {
	const shared_ptr<ImagePyramidLayer> layer = getLayer(width);
	return layer ? layer->getIndex() : -1;
}

double CellBasedPyramidFeatureExtractor::getMinScaleFactor() const {
	return pyramid->getMinScaleFactor();
}

double CellBasedPyramidFeatureExtractor::getMaxScaleFactor() const {
	return pyramid->getMaxScaleFactor();
}

double CellBasedPyramidFeatureExtractor::getIncrementalScaleFactor() const {
	return pyramid->getIncrementalScaleFactor();
}

Size CellBasedPyramidFeatureExtractor::getPatchSize() const {
	return Size(cellColumnCount, cellRowCount);
}

Size CellBasedPyramidFeatureExtractor::getImageSize() const {
	return pyramid->getImageSize();
}

vector<pair<int, double>> CellBasedPyramidFeatureExtractor::getLayerScales() const {
	vector<pair<int, double>> scales = pyramid->getLayerScales();
	for (pair<int, double>& scale : scales)
		scale.second /= cellSize;
	return scales;
}

vector<Size> CellBasedPyramidFeatureExtractor::getLayerSizes() const {
	return pyramid->getLayerSizes();
}

vector<Size> CellBasedPyramidFeatureExtractor::getPatchSizes() const {
	const vector<shared_ptr<ImagePyramidLayer>>& layers = pyramid->getLayers();
	vector<Size> sizes;
	sizes.reserve(layers.size());
	for (const auto& layer : layers)
		sizes.push_back(Size(layer->getOriginal(cellSize * cellColumnCount), layer->getOriginal(cellSize * cellRowCount)));
	return sizes;
}

const shared_ptr<ImagePyramidLayer> CellBasedPyramidFeatureExtractor::getLayer(size_t width) const {
	double scaleFactor = static_cast<double>(cellSize * cellColumnCount) / static_cast<double>(width);
	return pyramid->getLayer(scaleFactor);
}

} /* namespace imageprocessing */
