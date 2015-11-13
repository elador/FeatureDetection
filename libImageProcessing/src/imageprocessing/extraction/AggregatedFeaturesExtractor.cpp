/*
 * AggregatedFeaturesExtractor.cpp
 *
 *  Created on: 30.10.2015
 *      Author: poschmann
 */

#include "imageprocessing/extraction/AggregatedFeaturesExtractor.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"

using cv::Mat;
using cv::Point;
using cv::Point_;
using cv::Rect;
using cv::Size;
using std::make_shared;
using std::round;
using std::shared_ptr;

namespace imageprocessing {
namespace extraction {

AggregatedFeaturesExtractor::AggregatedFeaturesExtractor(shared_ptr<ImageFilter> layerFilter,
		Size patchSizeInCells, int cellSizeInPixels, int octaveLayerCount) :
				patchSizeInCells(patchSizeInCells),
				patchSizeInPixels(patchSizeInCells.width * cellSizeInPixels, patchSizeInCells.height * cellSizeInPixels),
				cellSizeInPixels(cellSizeInPixels),
				adjustMinScaleFactor(true) {
	// TODO get patch and cell size from AggregationFilter somehow (or create AggregationFilter here?)
	double minScaleFactor = 0.5; // will be reset when the image size is known
	double maxScaleFactor = 1; // detect up to the smallest possible target size
	featurePyramid = make_shared<ImagePyramid>(static_cast<size_t>(octaveLayerCount), minScaleFactor, maxScaleFactor);
	featurePyramid->addLayerFilter(layerFilter);
}

AggregatedFeaturesExtractor::AggregatedFeaturesExtractor(shared_ptr<ImageFilter> imageFilter, shared_ptr<ImageFilter> layerFilter,
		Size patchSizeInCells, int cellSizeInPixels, int octaveLayerCount) :
				AggregatedFeaturesExtractor(layerFilter, patchSizeInCells, cellSizeInPixels, octaveLayerCount) {
	featurePyramid->addImageFilter(imageFilter);
}

shared_ptr<ImagePyramid> AggregatedFeaturesExtractor::getFeaturePyramid() {
	return featurePyramid;
}

void AggregatedFeaturesExtractor::update(shared_ptr<VersionedImage> image) {
	if (adjustMinScaleFactor)
		featurePyramid->setMinScaleFactor(getMinScaleFactor(image->getData()));
	featurePyramid->update(image);
}

double AggregatedFeaturesExtractor::getMinScaleFactor(const Mat& image) {
	double minScaleFactor = static_cast<double>(patchSizeInPixels.width) / getMaxWidth(image);
	int maxLayerIndex = static_cast<int>(std::log(minScaleFactor) / std::log(featurePyramid->getIncrementalScaleFactor()));
	return std::pow(featurePyramid->getIncrementalScaleFactor(), maxLayerIndex);
}

int AggregatedFeaturesExtractor::getMaxWidth(const Mat& image) {
	double aspectRatio = static_cast<double>(patchSizeInPixels.height) / static_cast<double>(patchSizeInPixels.width);
	double imageAspectRatio = static_cast<double>(image.rows) / static_cast<double>(image.cols);
	if (aspectRatio > imageAspectRatio) // height is the limiting factor when rescaling
		return static_cast<int>(image.rows / aspectRatio);
	else // width is the limiting factor when rescaling
		return image.cols;
}

shared_ptr<Patch> AggregatedFeaturesExtractor::extract(int centerX, int centerY, int width, int height) const {
	Rect boundsInImagePixels = Patch::computeBounds(Point(centerX, centerY), Size(width, height));
	return extract(boundsInImagePixels);
}

shared_ptr<Patch> AggregatedFeaturesExtractor::extract(Rect bounds) const {
	const shared_ptr<ImagePyramidLayer> layer = getLayer(bounds.width);
	if (!layer)
		return shared_ptr<Patch>();
	Point_<double> centerInImagePixels(bounds.x + 0.5 * bounds.width, bounds.y + 0.5 * bounds.height);
	Point centerInLayerCells = computePointInLayerCells(centerInImagePixels, *layer);
	return extract(*layer, Patch::computeBounds(centerInLayerCells, patchSizeInCells));
}

const shared_ptr<ImagePyramidLayer> AggregatedFeaturesExtractor::getLayer(int width) const {
	double scaleFactor = static_cast<double>(patchSizeInPixels.width) / static_cast<double>(width);
	return featurePyramid->getLayer(scaleFactor);
}

Point AggregatedFeaturesExtractor::computePointInLayerCells(Point_<double> pointInImagePixels, const ImagePyramidLayer& layer) const {
	return Point(
			static_cast<int>(layer.getScaledX(pointInImagePixels.x) / cellSizeInPixels),
			static_cast<int>(layer.getScaledY(pointInImagePixels.y) / cellSizeInPixels)
	);
}

shared_ptr<Patch> AggregatedFeaturesExtractor::extract(const ImagePyramidLayer& layer, Rect boundsInLayerCells) const {
	const Mat& layerCellImage = layer.getScaledImage();
	if (!isPatchWithinImage(boundsInLayerCells, layerCellImage))
		return shared_ptr<Patch>();
	Mat data(layerCellImage, boundsInLayerCells);
	Rect boundsInImagePixels = computeBoundsInImagePixels(boundsInLayerCells, layer);
	return make_shared<Patch>(boundsInImagePixels, data.clone());
}

bool AggregatedFeaturesExtractor::isPatchWithinImage(Rect bounds, const Mat& image) const {
	return bounds.x >= 0
			&& bounds.y >= 0
			&& bounds.x + bounds.width <= image.cols
			&& bounds.y + bounds.height <= image.rows;
}

Rect AggregatedFeaturesExtractor::computeBoundsInImagePixels(Rect boundsInLayerCells, const ImagePyramidLayer& layer) const {
	return Rect(
			static_cast<int>(round(layer.getOriginalX(boundsInLayerCells.x * cellSizeInPixels))),
			static_cast<int>(round(layer.getOriginalY(boundsInLayerCells.y * cellSizeInPixels))),
			static_cast<int>(round(layer.getOriginalX(boundsInLayerCells.width * cellSizeInPixels))),
			static_cast<int>(round(layer.getOriginalY(boundsInLayerCells.height * cellSizeInPixels)))
	);
}

} /* namespace extraction */
} /* namespace imageprocessing */
