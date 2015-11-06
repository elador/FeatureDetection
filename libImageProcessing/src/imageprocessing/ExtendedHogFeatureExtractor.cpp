/*
 * ExtendedHogFeatureExtractor.cpp
 *
 *  Created on: 28.03.2014
 *      Author: poschmann
 */

#include "imageprocessing/ExtendedHogFeatureExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/GradientFilter.hpp"
#include "imageprocessing/GradientBinningFilter.hpp"
#include "imageprocessing/ExtendedHogFilter.hpp"
#include "imageprocessing/CompleteExtendedHogFilter.hpp"
#include "imageprocessing/Patch.hpp"
#include <stdexcept>

using cv::Mat;
using cv::Rect;
using cv::Point;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::invalid_argument;
using std::runtime_error;

namespace imageprocessing {

shared_ptr<ImagePyramid> ExtendedHogFeatureExtractor::createPyramid(int width, int minWidth, int maxWidth, int octaveLayerCount) {
	double incrementalScaleFactor = pow(0.5, 1. / octaveLayerCount);
	double minScaleFactor = static_cast<double>(width) / maxWidth;
	double maxScaleFactor = static_cast<double>(width) / minWidth;
	int maxLayerIndex = cvRound(log(minScaleFactor) / log(incrementalScaleFactor));
	int minLayerIndex = cvRound(log(maxScaleFactor) / log(incrementalScaleFactor));
	maxScaleFactor = pow(incrementalScaleFactor, minLayerIndex);
	minScaleFactor = pow(incrementalScaleFactor, maxLayerIndex);
	return make_shared<ImagePyramid>(static_cast<size_t>(octaveLayerCount), minScaleFactor, maxScaleFactor);
}

ExtendedHogFeatureExtractor::ExtendedHogFeatureExtractor(shared_ptr<ImagePyramid> pyramid,
		shared_ptr<CompleteExtendedHogFilter> ehogFilter, int cols, int rows) :
				pyramid(pyramid), ehogFilter(ehogFilter), patchWidth((cols + 2) * ehogFilter->getCellSize()), patchHeight((rows + 2) * ehogFilter->getCellSize()),
				cellSize(ehogFilter->getCellSize()), widthFactor(static_cast<double>(cols + 2) / cols), heightFactor(static_cast<double>(rows + 2) / rows) {
	if (cols <= 0 || rows <= 0)
		throw invalid_argument("ExtendedHogFeatureExtractor: the amount of columns and rows must be greater than zero");
}

ExtendedHogFeatureExtractor::ExtendedHogFeatureExtractor(shared_ptr<ImagePyramid> pyramid,
		shared_ptr<ExtendedHogFilter> ehogFilter, int cols, int rows) :
				pyramid(pyramid), ehogFilter(ehogFilter), patchWidth((cols + 2) * ehogFilter->getCellWidth()), patchHeight((rows + 2) * ehogFilter->getCellWidth()),
				cellSize(ehogFilter->getCellWidth()), widthFactor(static_cast<double>(cols + 2) / cols), heightFactor(static_cast<double>(rows + 2) / rows) {
	if (cols <= 0 || rows <= 0)
		throw invalid_argument("ExtendedHogFeatureExtractor: the amount of columns and rows must be greater than zero");
	if (ehogFilter->getCellWidth() != ehogFilter->getCellHeight())
		throw invalid_argument("ExtendedHogFeatureExtractor: the cell width and height have to be the same");
}

ExtendedHogFeatureExtractor::ExtendedHogFeatureExtractor(shared_ptr<GradientFilter> gradientFilter,
		shared_ptr<GradientBinningFilter> binningFilter, shared_ptr<ExtendedHogFilter> ehogFilter,
		int cols, int rows, int minWidth, int maxWidth, int octaveLayerCount) :
				pyramid(createPyramid((cols + 2) * ehogFilter->getCellWidth(), (cols + 2) * minWidth / cols, (cols + 2) * maxWidth / cols, octaveLayerCount)),
				ehogFilter(ehogFilter), patchWidth((cols + 2) * ehogFilter->getCellWidth()), patchHeight((rows + 2) * ehogFilter->getCellWidth()),
				cellSize(ehogFilter->getCellWidth()), widthFactor(static_cast<double>(cols + 2) / cols), heightFactor(static_cast<double>(rows + 2) / rows) {
	if (cols <= 0 || rows <= 0)
		throw invalid_argument("ExtendedHogFeatureExtractor: the amount of columns and rows must be greater than zero");
	if (ehogFilter->getCellWidth() != ehogFilter->getCellHeight())
		throw invalid_argument("ExtendedHogFeatureExtractor: the cell width and height have to be the same");
	pyramid->addImageFilter(make_shared<GrayscaleFilter>());
	pyramid->addLayerFilter(gradientFilter);
	pyramid->addLayerFilter(binningFilter);
}

ExtendedHogFeatureExtractor::ExtendedHogFeatureExtractor(shared_ptr<CompleteExtendedHogFilter> ehogFilter,
		int cols, int rows, int minWidth, int maxWidth, int octaveLayerCount) :
				pyramid(createPyramid((cols + 2) * ehogFilter->getCellSize(), (cols + 2) * minWidth / cols, (cols + 2) * maxWidth / cols, octaveLayerCount)),
				ehogFilter(ehogFilter), patchWidth((cols + 2) * ehogFilter->getCellSize()), patchHeight((rows + 2) * ehogFilter->getCellSize()),
				cellSize(ehogFilter->getCellSize()), widthFactor(static_cast<double>(cols + 2) / cols), heightFactor(static_cast<double>(rows + 2) / rows) {
	if (cols <= 0 || rows <= 0)
		throw invalid_argument("ExtendedHogFeatureExtractor: the amount of columns and rows must be greater than zero");
	pyramid->addImageFilter(make_shared<GrayscaleFilter>());
}

ExtendedHogFeatureExtractor::ExtendedHogFeatureExtractor(const ExtendedHogFeatureExtractor& other) :
		pyramid(make_shared<ImagePyramid>(*other.pyramid)), ehogFilter(other.ehogFilter),
		patchWidth(other.patchWidth), patchHeight(other.patchHeight), cellSize(other.cellSize),
		widthFactor(other.widthFactor), heightFactor(other.heightFactor) {}

void ExtendedHogFeatureExtractor::update(shared_ptr<VersionedImage> image) {
	pyramid->update(image);
}

shared_ptr<Patch> ExtendedHogFeatureExtractor::extract(int x, int y, int width, int height) const {
	width = static_cast<int>(std::round(widthFactor * width));
	height = static_cast<int>(std::round(heightFactor * height));
	double scaleFactor = static_cast<double>(patchWidth) / static_cast<double>(width);
	const shared_ptr<ImagePyramidLayer> layer = pyramid->getLayer(scaleFactor);
	if (!layer)
		return shared_ptr<Patch>();

	const Mat& image = layer->getScaledImage();
	Rect bounds(layer->getScaled(x - width / 2), layer->getScaled(y - height / 2), patchWidth, patchHeight);
	if (bounds.x < -cellSize || bounds.x + bounds.width > image.cols + cellSize
			|| bounds.y < -cellSize || bounds.y + bounds.height > image.rows + cellSize)
		return shared_ptr<Patch>();
	Mat patchData;
	if (bounds.x >= 0 && bounds.y >= 0 && bounds.x + bounds.width <= image.cols && bounds.y + bounds.height <= image.rows) {
		patchData = Mat(image, bounds);
	} else { // patch is partially outside the image
		vector<int> rowIndices = createIndexLut(image.rows, bounds.y, bounds.height);
		vector<int> colIndices = createIndexLut(image.cols, bounds.x, bounds.width);
		if (image.type() == CV_8UC1)
			patchData = createPatchData<uchar>(image, rowIndices, colIndices);
		else if (image.type() == CV_8UC2)
			patchData = createPatchData<cv::Vec2b>(image, rowIndices, colIndices);
		else if (image.type() == CV_8UC4)
			patchData = createPatchData<cv::Vec4b>(image, rowIndices, colIndices);
		else
			throw runtime_error("ExtendedHogFeatureExtractor: the type of the pyramid layer images has to be CV_8UC1, CV_8UC2 or CV_8UC4");
	}
	Mat tmp = ehogFilter->applyTo(patchData);
	Mat data = Mat(tmp, Rect(1, 1, tmp.cols - 2, tmp.rows - 2)).clone();
	int originalWidth = layer->getOriginal(bounds.width - 2 * cellSize);
	int originalHeight = layer->getOriginal(bounds.height - 2 * cellSize);
	int originalX = layer->getOriginal(bounds.x + cellSize) + originalWidth / 2;
	int originalY = layer->getOriginal(bounds.y + cellSize) + originalHeight / 2;
	return make_shared<Patch>(originalX, originalY, originalWidth, originalHeight, data);
}

vector<int> ExtendedHogFeatureExtractor::createIndexLut(int imageSize, int patchStart, int patchSize) const {
	vector<int> indices(patchSize);
	for (int patchIndex = 0; patchIndex < patchSize; ++patchIndex) {
		int imageIndex = patchStart + patchIndex;
		if (imageIndex < 0)
			imageIndex = -imageIndex - 1;
		else if (imageIndex >= imageSize)
			imageIndex = 2 * imageSize - imageIndex - 1;
		indices[patchIndex] = imageIndex;
	}
	return indices;
}

shared_ptr<ImagePyramid> ExtendedHogFeatureExtractor::getPyramid() {
	return pyramid;
}

const shared_ptr<ImagePyramid> ExtendedHogFeatureExtractor::getPyramid() const {
	return pyramid;
}

int ExtendedHogFeatureExtractor::getPatchWidth() const {
	return patchWidth;
}

int ExtendedHogFeatureExtractor::getPatchHeight() const {
	return patchHeight;
}

} /* namespace imageprocessing */
