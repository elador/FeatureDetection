/*
 * DirectPyramidFeatureExtractor.cpp
 *
 *  Created on: 22.03.2013
 *      Author: poschmann
 */

#include "HogExtractor.hpp"
#include "imageprocessing/Patch.hpp"
#include "boost/iterator/indirect_iterator.hpp"

using cv::Point;
using boost::make_indirect_iterator;
using std::make_shared;

namespace imageprocessing {

HogExtractor::HogExtractor(shared_ptr<IntegralHistogramPyramid> pyramid, int width, int height,
		int cellSize, int blockSize, bool combineHistograms, Normalization normalization) :
		HistogramFeatureExtractor(normalization),
		pyramid(pyramid), patchWidth(width), patchHeight(height),
		cellWidth(cellSize),
		cellHeight(cellSize),
		blockWidth(blockSize),
		blockHeight(blockSize),
		combineHistograms(combineHistograms) {}

HogExtractor::HogExtractor(int width, int height, int minWidth, int maxWidth, double incrementalScaleFactor,
		int cellSize, int blockSize, bool combineHistograms, Normalization normalization) :
		HistogramFeatureExtractor(normalization),
		pyramid(make_shared<IntegralHistogramPyramid>(static_cast<double>(width) / maxWidth, static_cast<double>(width) / minWidth, incrementalScaleFactor)),
		patchWidth(width), patchHeight(height),
		cellWidth(cellWidth),
		cellHeight(cellHeight),
		blockWidth(blockWidth),
		blockHeight(blockHeight),
		combineHistograms(combineHistograms) {}

HogExtractor::~HogExtractor() {}

vector<Size> HogExtractor::getPatchSizes() const {
	const vector<shared_ptr<IntegralHistogramPyramidLayer>>& layers = pyramid->getLayers();
	vector<Size> sizes;
	sizes.resize(layers.size());
	for (auto layer = make_indirect_iterator(layers.begin()); layer != make_indirect_iterator(layers.end()); ++layer)
		sizes.push_back(Size(layer->getOriginal(patchWidth), layer->getOriginal(patchHeight)));
	return sizes;
}

shared_ptr<Patch> HogExtractor::extract(int x, int y, int width, int height) const {
	const shared_ptr<IntegralHistogramPyramidLayer> layer = getLayer(width);
	if (!layer)
		return shared_ptr<Patch>();
	int scaledX = layer->getScaled(x);
	int scaledY = layer->getScaled(y);
	int patchBeginX = scaledX - patchWidth / 2; // inclusive
	int patchBeginY = scaledY - patchHeight / 2; // inclusive
	int patchEndX = patchBeginX + patchWidth; // exclusive
	int patchEndY = patchBeginY + patchHeight; // exclusive
	if (patchBeginX < 0 || patchEndX > layer->getSize().width
			|| patchBeginY < 0 || patchEndY > layer->getSize().height)
		return shared_ptr<Patch>();
	int originalX = layer->getOriginal(scaledX);
	int originalY = layer->getOriginal(scaledY);
	int originalWidth = layer->getOriginal(patchWidth);
	int originalHeight = layer->getOriginal(patchHeight);
	Rect patchBounds(patchBeginX, patchBeginY, patchWidth, patchHeight);
	const vector<Mat>& image = layer->getScaledImage();
	vector<Mat> data;
	data.reserve(image.size());
	for (auto binData = image.begin(); binData != image.end(); ++binData)
		data.push_back(Mat(*binData, patchBounds));
	return make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, getHogFeature(data));
}

vector<shared_ptr<Patch>> HogExtractor::extract(int stepX, int stepY, Rect roi, int firstLayer, int lastLayer) const {
	if (roi.x == 0 && roi.y == 0 && roi.width == 0 && roi.height == 0) {
		Size size = getImageSize();
		roi.width = size.width;
		roi.height = size.height;
	}
	vector<shared_ptr<Patch>> patches;
	const vector<shared_ptr<IntegralHistogramPyramidLayer>>& layers = pyramid->getLayers();
	if (firstLayer < 0)
		firstLayer = layers.front()->getIndex();
	if (lastLayer < 0)
		lastLayer = layers.back()->getIndex();
	for (auto layIt = layers.begin(); layIt != layers.end(); ++layIt) {
		shared_ptr<IntegralHistogramPyramidLayer> layer = *layIt;
		if (layer->getIndex() < firstLayer)
			continue;
		if (layer->getIndex() > lastLayer)
			break;

		int originalWidth = layer->getOriginal(patchWidth);
		int originalHeight = layer->getOriginal(patchHeight);
		const vector<Mat>& image = layer->getScaledImage();

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
				vector<Mat> data;
				data.reserve(image.size());
				for (auto binData = image.begin(); binData != image.end(); ++binData)
					data.push_back(Mat(*binData, patchBounds));
				int originalX = layer->getOriginal(center.x);
				int originalY = layer->getOriginal(center.y);
				patches.push_back(make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, getHogFeature(data)));
				patchBounds.x += stepX;
				center.x += stepX;
			}
			patchBounds.y += stepY;
			center.y += stepY;
		}
	}
	return patches;
}

shared_ptr<Patch> HogExtractor::extract(int layerIndex, int x, int y) const {
	shared_ptr<IntegralHistogramPyramidLayer> layer = pyramid->getLayer(layerIndex);
	if (!layer)
		return shared_ptr<Patch>();
	Rect patchBounds(x - patchWidth / 2, y - patchHeight / 2, patchWidth, patchHeight);
	if (patchBounds.x < 0 || patchBounds.y < 0
			|| patchBounds.x + patchBounds.width >= layer->getSize().width
			|| patchBounds.y + patchBounds.height >= layer->getSize().height)
		return shared_ptr<Patch>();
	int originalX = layer->getOriginal(x);
	int originalY = layer->getOriginal(y);
	int originalWidth = layer->getOriginal(patchWidth);
	int originalHeight = layer->getOriginal(patchHeight);
	const vector<Mat>& image = layer->getScaledImage();
	vector<Mat> data;
	data.reserve(image.size());
	for (auto binData = image.begin(); binData != image.end(); ++binData)
		data.push_back(Mat(*binData, patchBounds));
	return make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, getHogFeature(data));
}

Mat HogExtractor::getHogFeature(const vector<Mat>& integralHistogram) const {
	unsigned int bins = integralHistogram.size();
	// create histograms of cells
	int cellRows = cvRound(static_cast<double>(patchHeight) / static_cast<double>(cellHeight));
	int cellCols = cvRound(static_cast<double>(patchWidth) / static_cast<double>(cellWidth));
	vector<Mat> cellHistograms(cellRows * cellCols);
	for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
		for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
			Mat& histogram = cellHistograms[cellRow * cellCols + cellCol];
			histogram = Mat::zeros(1, bins, CV_32F);
			float* histogramValues = histogram.ptr<float>(0);
			int startRow = (cellRow * patchHeight) / cellRows;
			int startCol = (cellCol * patchWidth) / cellCols;
			int endRow = ((cellRow + 1) * patchHeight) / cellRows;
			int endCol = ((cellCol + 1) * patchWidth) / cellCols;
			for (int bin = 0; bin < integralHistogram.size(); ++bin) {
				const Mat& binData = integralHistogram[bin];
				histogramValues[bin] =
						binData.at<float>(startRow, startCol)
						+ binData.at<float>(endRow, endCol)
						- binData.at<float>(startRow, endCol)
						- binData.at<float>(endRow, startCol);
			}
		}
	}

	// create histograms of blocks
	int blockHistogramSize;
	int blockRows = cellRows - blockHeight + 1;
	int blockCols = cellCols - blockWidth + 1;
	vector<Mat> blockHistograms(blockRows * blockCols);
	for (int blockRow = 0; blockRow < blockRows; ++blockRow) {
		for (int blockCol = 0; blockCol < blockCols; ++blockCol) {
			Mat& blockHistogram = blockHistograms[blockRow * blockCols + blockCol];
			if (combineHistograms) { // combine histograms by adding the bin values
				blockHistogramSize = bins;
				blockHistogram = Mat::zeros(1, bins, CV_32F);
				float* blockHistogramValues = blockHistogram.ptr<float>(0);
				for (int cellRow = blockRow; cellRow < blockRow + blockHeight; ++cellRow) {
					for (int cellCol = blockCol; cellCol < blockCol + blockWidth; ++cellCol) {
						Mat& cellHistogram = cellHistograms[blockRow * blockCols + blockCol];
						float* cellHistogramValues = cellHistogram.ptr<float>(0);
						for (unsigned int bin = 0; bin < bins; ++bin)
							blockHistogramValues[bin] += cellHistogramValues[bin];
					}
				}
			} else { // create concatenation of histograms
				blockHistogramSize = blockWidth * blockHeight * bins;
				blockHistogram.create(1, blockHistogramSize, CV_32F);
				float* blockHistogramValues = blockHistogram.ptr<float>(0);
				for (int cellRow = blockRow; cellRow < blockRow + blockHeight; ++cellRow) {
					for (int cellCol = blockCol; cellCol < blockCol + blockWidth; ++cellCol) {
						Mat& cellHistogram = cellHistograms[blockRow * blockCols + blockCol];
						float* cellHistogramValues = cellHistogram.ptr<float>(0);
						for (unsigned int bin = 0; bin < bins; ++bin)
							blockHistogramValues[bin] = cellHistogramValues[bin];
						blockHistogramValues += bins;
					}
				}
			}

			normalize(blockHistogram);
		}
	}
	Mat patchData(1, blockHistograms.size() * blockHistogramSize, CV_32F);
	float* patchValues = patchData.ptr<float>(0);
	for (unsigned int histIndex = 0; histIndex < blockHistograms.size(); ++histIndex) {
		Mat& blockHistogram = blockHistograms[histIndex];
		float* blockHistogramValues = blockHistogram.ptr<float>(0);
		for (int bin = 0; bin < blockHistogramSize; ++bin)
			patchValues[bin] = blockHistogramValues[bin];
		patchValues += blockHistogramSize;
	}

	image = Mat::zeros(20 * cellRows, 20 * cellCols, CV_8U);
	for (int cellRow = 0; cellRow < cellRows; ++cellRow) {
		for (int cellCol = 0; cellCol < cellCols; ++cellCol) {
			Mat& histogram = cellHistograms[cellRow * cellCols + cellCol];
			for (int i = 0; i < histogram.cols; ++i) {
				double angle = (i + 0.5) * CV_PI / bins;
				double x = cos(angle);
				double y = sin(angle);
				double xStart = 10 * x + 10 + 20 * cellCol;
				double xEnd = -10 * x + 10 + 20 * cellCol;
				double yStart = 10 * y + 10 + 20 * cellRow;
				double yEnd = -10 * y + 10 + 20 * cellRow;
				float weight = histogram.at<float>(i) / 20;
				cv::Scalar color(255 * weight, 255 * weight, 255 * weight);
				cv::line(image, cv::Point(xStart, yStart), cv::Point(xEnd, yEnd), color);
			}
		}
	}

	return patchData;
}

} /* namespace imageprocessing */
