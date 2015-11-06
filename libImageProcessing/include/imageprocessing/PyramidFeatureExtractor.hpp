/*
 * PyramidFeatureExtractor.hpp
 *
 *  Created on: 18.02.2013
 *      Author: poschmann
 */
#pragma once

#ifndef PYRAMIDFEATUREEXTRACTOR_HPP_
#define PYRAMIDFEATUREEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include <vector>
#include <utility>

namespace imageprocessing {

/**
 * Feature extractor whose features are patches of a constant size extracted from an image pyramid.
 */
class PyramidFeatureExtractor : public FeatureExtractor {
public:

	virtual ~PyramidFeatureExtractor() {}

	using FeatureExtractor::update;

	virtual void update(std::shared_ptr<VersionedImage> image) = 0;

	/**
	 * Extracts a patch from the corresponding image pyramid.
	 *
	 * @param[in] x The x-coordinate of the patch center in the original image.
	 * @param[in] y The y-coordinate of the patch center in the original image.
	 * @param[in] width The width of the patch in the original image.
	 * @param[in] height The height of the patch in the original image.
	 * @return The extracted patch or an empty pointer in case the patch could not be extracted.
	 */
	virtual std::shared_ptr<Patch> extract(int x, int y, int width, int height) const = 0;

	/**
	 * Extracts several patches from the layers of the corresponding image pyramid.
	 *
	 * @param[in] stepX The step size in x-direction in pixels (will be the same absolute value in all pyramid layers).
	 * @param[in] stepY The step size in y-direction in pixels (will be the same absolute value in all pyramid layers).
	 * @param[in] roi The region of interest inside the original image (region will be scaled accordingly to the layers).
	 * @param[in] firstLayer The index of the first layer to extract patches from.
	 * @param[in] lastLayer The index of the last layer to extract patches from.
	 * @param[in] stepLayer The step size for proceeding to the next layer (values greater than one will skip layers).
	 * @return The extracted patches.
	 */
	virtual std::vector<std::shared_ptr<Patch>> extract(int stepX, int stepY, cv::Rect roi = cv::Rect(),
			int firstLayer = -1, int lastLayer = -1, int stepLayer = 1) const = 0;

	/**
	 * Extracts a single patch from a layer of the corresponding image pyramid.
	 *
	 * @param[in] layer The index of the layer.
	 * @param[in] x The x-coordinate of the patch center inside the layer.
	 * @param[in] y The y-coordinate of the patch center inside the layer.
	 * @return The extracted patch or an empty pointer in case the patch could not be extracted.
	 */
	virtual std::shared_ptr<Patch> extract(int layer, int x, int y) const = 0;

	/**
	 * Determines the index of the pyramid layer that approximately contains patches of the given size.
	 *
	 * @param[in] width The width of the patches.
	 * @param[in] height The height of the patches.
	 * @return The index of the pyramid layer or -1 if there is no layer with an appropriate patch size.
	 */
	virtual int getLayerIndex(int width, int height) const = 0;

	/**
	 * @return The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 */
	virtual double getMinScaleFactor() const = 0;

	/**
	 * @return The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 */
	virtual double getMaxScaleFactor() const = 0;

	/**
	 * @return The incremental scale factor between two layers of the pyramid.
	 */
	virtual double getIncrementalScaleFactor() const = 0;

	/**
	 * @return The size of the extracted patches.
	 */
	virtual cv::Size getPatchSize() const = 0;

	/**
	 * @return The size of the original image.
	 */
	virtual cv::Size getImageSize() const = 0;

	/**
	 * Determines the scale factors of each pyramid layer.
	 *
	 * @return Pairs containing the index and scale factor of each pyramid layer, beginning from the largest layer.
	 */
	virtual std::vector<std::pair<int, double>> getLayerScales() const = 0;

	/**
	 * Determines the size of the scaled image of each pyramid layer.
	 *
	 * @return The sizes of the pyramid layer's scaled images, beginning from the largest layer.
	 */
	virtual std::vector<cv::Size> getLayerSizes() const = 0;

	/**
	 * Determines the size of the patch of each pyramid layer when scaled to the original image.
	 *
	 * @return The sizes of the pyramid layer's patches, beginning from the smallest patch (largest layer).
	 */
	virtual std::vector<cv::Size> getPatchSizes() const = 0;
};

} /* namespace imageprocessing */
#endif /* PYRAMIDFEATUREEXTRACTOR_HPP_ */
