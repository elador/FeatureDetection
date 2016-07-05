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

AggregatedFeaturesExtractor::AggregatedFeaturesExtractor(shared_ptr<ImagePyramid> featurePyramid,
		Size patchSizeInCells, int cellSizeInPixels, bool adjustMinScaleFactor, int minPatchWidthInPixels) :
				featurePyramid(featurePyramid),
				patchSizeInCells(patchSizeInCells),
				patchSizeInPixels(patchSizeInCells * cellSizeInPixels),
				cellSizeInPixels(cellSizeInPixels),
				adjustMinScaleFactor(adjustMinScaleFactor) {
	if (minPatchWidthInPixels > patchSizeInPixels.width)
		featurePyramid->setMaxScaleFactor(getMaxScaleFactor(minPatchWidthInPixels));
}

AggregatedFeaturesExtractor::AggregatedFeaturesExtractor(shared_ptr<ImageFilter> layerFilter,
		Size patchSizeInCells, int cellSizeInPixels, int octaveLayerCount, int minPatchWidthInPixels) :
				AggregatedFeaturesExtractor(make_shared<ImagePyramid>(static_cast<size_t>(octaveLayerCount), 0.5, 1),
						patchSizeInCells, cellSizeInPixels, true, minPatchWidthInPixels) {
	featurePyramid->addLayerFilter(layerFilter);
}

AggregatedFeaturesExtractor::AggregatedFeaturesExtractor(shared_ptr<ImageFilter> imageFilter, shared_ptr<ImageFilter> layerFilter,
		Size patchSizeInCells, int cellSizeInPixels, int octaveLayerCount, int minPatchWidthInPixels) :
				AggregatedFeaturesExtractor(layerFilter, patchSizeInCells, cellSizeInPixels, octaveLayerCount, minPatchWidthInPixels) {
	featurePyramid->addImageFilter(imageFilter);
}

double AggregatedFeaturesExtractor::getMaxScaleFactor(int minPatchWidthInPixels) const {
	assert(minPatchWidthInPixels > patchSizeInPixels.width);
	double maxScaleFactor = static_cast<double>(patchSizeInPixels.width) / minPatchWidthInPixels;
	int minLayerIndex = static_cast<int>(std::ceil(std::log(maxScaleFactor) / std::log(featurePyramid->getIncrementalScaleFactor())));
	return std::pow(featurePyramid->getIncrementalScaleFactor(), minLayerIndex);
}

shared_ptr<ImagePyramid> AggregatedFeaturesExtractor::getFeaturePyramid() {
	return featurePyramid;
}

void AggregatedFeaturesExtractor::update(shared_ptr<VersionedImage> image) {
	if (adjustMinScaleFactor)
		featurePyramid->setMinScaleFactor(getMinScaleFactor(image->getData()));
	featurePyramid->update(image);
}

double AggregatedFeaturesExtractor::getMinScaleFactor(const Mat& image) const {
	double minScaleFactor = static_cast<double>(patchSizeInPixels.width) / getMaxWidth(image);
	int maxLayerIndex = static_cast<int>(std::log(minScaleFactor) / std::log(featurePyramid->getIncrementalScaleFactor()));
	return std::pow(featurePyramid->getIncrementalScaleFactor(), maxLayerIndex);
}

int AggregatedFeaturesExtractor::getMaxWidth(const Mat& image) const {
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
