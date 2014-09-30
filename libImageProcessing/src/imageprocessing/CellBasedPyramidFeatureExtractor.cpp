/*
 * CellBasedPyramidFeatureExtractor.cpp
 *
 *  Created on: 16.12.2013
 *      Author: poschmann
 */

#include "imageprocessing/CellBasedPyramidFeatureExtractor.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include <stdexcept>

using cv::Mat;
using std::shared_ptr;
using std::invalid_argument;

namespace imageprocessing {

CellBasedPyramidFeatureExtractor::CellBasedPyramidFeatureExtractor(
		int cellSize, int cols, int rows, int minWidth, int maxWidth, int octaveLayerCount) :
				PyramidFeatureExtractor(createPyramid(cols * cellSize, minWidth, maxWidth, octaveLayerCount), cols, rows),
				cellSize(cellSize), realPatchWidth(cols * cellSize) {
	if (cellSize <= 0)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: the cell size must be greater than zero");
	if (cols <= 0 || rows <= 0)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: the amount of columns and rows must be greater than zero");
}

CellBasedPyramidFeatureExtractor::CellBasedPyramidFeatureExtractor(
		shared_ptr<ImagePyramid> pyramid, int cellSize, int cols, int rows) :
				PyramidFeatureExtractor(pyramid, cols, rows), cellSize(cellSize), realPatchWidth(cols * cellSize) {
	if (cellSize <= 0)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: the cell size must be greater than zero");
	if (cols <= 0 || rows <= 0)
		throw invalid_argument("CellBasedPyramidFeatureExtractor: the amount of columns and rows must be greater than zero");
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

} /* namespace imageprocessing */
