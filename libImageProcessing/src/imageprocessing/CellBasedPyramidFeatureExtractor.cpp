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

using cv::Mat;
using cv::Rect;
using cv::Point;
using std::pair;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::invalid_argument;

namespace imageprocessing {

CellBasedPyramidFeatureExtractor::CellBasedPyramidFeatureExtractor(
		int cellSize, int cols, int rows, int minWidth, int maxWidth, int octaveLayerCount) :
				DirectPyramidFeatureExtractor(createPyramid(cols * cellSize, minWidth, maxWidth, octaveLayerCount), cols, rows),
				cellSize(cellSize), realPatchWidth(cols * cellSize) {
	if (cellSize <= 0)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: the cell size must be greater than zero");
	if (cols <= 0 || rows <= 0)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: the amount of columns and rows must be greater than zero");
}

CellBasedPyramidFeatureExtractor::CellBasedPyramidFeatureExtractor(
		shared_ptr<ImagePyramid> pyramid, int cellSize, int cols, int rows) :
				DirectPyramidFeatureExtractor(pyramid, cols, rows), cellSize(cellSize), realPatchWidth(cols * cellSize) {
	if (cellSize <= 0)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: the cell size must be greater than zero");
	if (cols <= 0 || rows <= 0)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: the amount of columns and rows must be greater than zero");
}

vector<shared_ptr<Patch>> CellBasedPyramidFeatureExtractor::extract(int stepX, int stepY, Rect roi,
		int firstLayer, int lastLayer, int stepLayer) const {
	if (stepX < 1)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: stepX has to be greater than zero");
	if (stepY < 1)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: stepY has to be greater than zero");
	return DirectPyramidFeatureExtractor::extract(
			std::max(1, static_cast<int>(std::round(stepX / static_cast<double>(cellSize)))),
			std::max(1, static_cast<int>(std::round(stepY / static_cast<double>(cellSize)))),
			roi, firstLayer, lastLayer, stepLayer);
}

const shared_ptr<ImagePyramidLayer> CellBasedPyramidFeatureExtractor::getLayer(int width) const {
	double scaleFactor = static_cast<double>(realPatchWidth) / static_cast<double>(width);
	return getPyramid()->getLayer(scaleFactor);
}

int CellBasedPyramidFeatureExtractor::getScaled(const ImagePyramidLayer& layer, int value) const {
	return cvRound(value * layer.getScaleFactor() / cellSize);
}

int CellBasedPyramidFeatureExtractor::getOriginal(const ImagePyramidLayer& layer, int value) const {
	return cvRound(value * cellSize / layer.getScaleFactor());
}

vector<pair<int, double>> CellBasedPyramidFeatureExtractor::getLayerScales() const {
	vector<pair<int, double>> scales = getPyramid()->getLayerScales();
	for (pair<int, double>& scale : scales)
		scale.second /= cellSize;
	return scales;
}

} /* namespace imageprocessing */
