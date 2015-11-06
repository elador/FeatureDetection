/*
 * ImagePyramid.hpp
 *
 *  Created on: 15.02.2013
 *      Author: huber & poschmann
 */

#pragma once

#ifndef IMAGEPYRAMID_HPP_
#define IMAGEPYRAMID_HPP_

#include "imageprocessing/Version.hpp"
#include "opencv2/core/core.hpp"
#include <vector>
#include <memory>
#include <utility>

namespace imageprocessing {

class VersionedImage;
class ImagePyramidLayer;
class ImageFilter;
class ChainedFilter;

/**
 * Image pyramid consisting of scaled representations of an image.
 */
class ImagePyramid {
public:

	/**
	 * Constructs a new empty image pyramid.
	 *
	 * @param[in] octaveLayerCount The number of layers per octave.
	 * @param[in] minScaleFactor The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 * @param[in] maxScaleFactor The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 */
	ImagePyramid(size_t octaveLayerCount, double minScaleFactor, double maxScaleFactor = 1);

	/**
	 * Constructs a new empty image pyramid using the incremental scale factor for determining the number of layers per octave.
	 *
	 * @param[in] incrementalScaleFactor The incremental scale factor between two layers of the pyramid.
	 * @param[in] minScaleFactor The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 * @param[in] maxScaleFactor The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 */
	ImagePyramid(double incrementalScaleFactor, double minScaleFactor, double maxScaleFactor = 1);

	/**
	 * Constructs a new empty image pyramid that will use another pyramid as its source.
	 */
	explicit ImagePyramid(double minScaleFactor = 0, double maxScaleFactor = 1);

	/**
	 * Constructs a new empty image pyramid with another pyramid as its source.
	 *
	 * @param[in] pyramid The new source pyramid.
	 * @param[in] minScaleFactor The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 * @param[in] maxScaleFactor The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 */
	explicit ImagePyramid(std::shared_ptr<ImagePyramid> pyramid, double minScaleFactor = 0, double maxScaleFactor = 1);

	/**
	 * Changes the source to the given image. The pyramid will not get updated, therefore for the source change to take
	 * effect update() has to be called. The next call of update() will force an update based on the new image.
	 *
	 * @param[in] image The new source image.
	 */
	void setSource(const cv::Mat& image);

	/**
	 * Changes the source to the given versioned image. The pyramid will not get updated, therefore for the source change
	 * to take effect update() has to be called. The next call of update() will only update this pyramid if the version of
	 * the given image differs from the one of this pyramid.
	 *
	 * @param[in] image The new source image.
	 */
	void setSource(const std::shared_ptr<VersionedImage>& image);

	/**
	 * Changes the source to the given pyramid. This pyramid will not get updated, therefore for the source change to take
	 * effect update() has to be called. The next call of update() will only update this pyramid if the version of the given
	 * pyramid differs from the one of this pyramid.
	 *
	 * @param[in] pyramid The new source pyramid.
	 */
	void setSource(const std::shared_ptr<ImagePyramid>& pyramid);

	/**
	 * Updates this pyramid using its source (either an image or another pyramid). If the source has not changed since the
	 * last update (version is the same), then this pyramid will not be changed.
	 */
	void update();

	/**
	 * Forces an update of this pyramid using the saved parameters. If this pyramid's source is another pyramid, then
	 * that pyramid will be updated first with the given image. If this pyramid's source is an image or it has no source
	 * yet, the image will be the new source.
	 *
	 * @param[in] image The new image.
	 */
	void update(const cv::Mat& image);

	/**
	 * Updates this pyramid using the saved parameters. If this pyramid's source is another pyramid, then that pyramid
	 * will be updated first with the given image. If this pyramid's source is an image or it has no source yet, the
	 * image will be the new source.
	 *
	 * @param[in] image The new image.
	 */
	void update(const std::shared_ptr<VersionedImage>& image);

	/**
	 * Adds a new filter that is applied to the original image after the currently existing image filters.
	 *
	 * @param[in] filter The new image filter.
	 */
	void addImageFilter(const std::shared_ptr<ImageFilter>& filter);

	/**
	 * Adds a new filter that is applied to the down-scaled images after the currently existing layer filters.
	 *
	 * @param[in] filter The new layer filter.
	 */
	void addLayerFilter(const std::shared_ptr<ImageFilter>& filter);

	/**
	 * Determines the pyramid layer with the given index.
	 *
	 * @param[in] index The index of the pyramid layer.
	 * @return The pointer to a pyramid layer that may be empty if there is no layer with the given index.
	 */
	const std::shared_ptr<ImagePyramidLayer> getLayer(int index) const;

	/**
	 * Determines the pyramid layer that is closest to the given scale factor.
	 *
	 * @param[in] scaleFactor The approximate scale factor of the layer.
	 * @return The pointer to a pyramid layer that may be empty if no layer has an appropriate scale factor.
	 */
	const std::shared_ptr<ImagePyramidLayer> getLayer(double scaleFactor) const;

	/**
	 * Determines the scale factors of each pyramid layer.
	 *
	 * @return Pairs containing the index and scale factor of each pyramid layer, beginning from the largest layer.
	 */
	std::vector<std::pair<int, double>> getLayerScales() const;

	/**
	 * Determines the size of the scaled image of each pyramid layer.
	 *
	 * @return The sizes of the pyramid layer's scaled images, beginning from the largest layer.
	 */
	std::vector<cv::Size> getLayerSizes() const;

	/**
	 * @return A reference to the pyramid layers.
	 */
	const std::vector<std::shared_ptr<ImagePyramidLayer>>& getLayers() const {
		return layers;
	}

	/**
	 * @return The number of layers per octave.
	 */
	size_t getOctaveLayerCount() const {
		return octaveLayerCount;
	}

	/**
	 * @return The incremental scale factor between two layers of the pyramid.
	 */
	double getIncrementalScaleFactor() const {
		return incrementalScaleFactor;
	}

	/**
	 * @return The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 */
	double getMinScaleFactor() const {
		return minScaleFactor;
	}

	/**
	 * Changes the minimum scale factor.
	 *
	 * @param[in] scaleFactor The new minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 */
	void setMinScaleFactor(double scaleFactor) {
		minScaleFactor = scaleFactor;
	}

	/**
	 * @return The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 */
	double getMaxScaleFactor() const {
		return maxScaleFactor;
	}

	/**
	 * Changes the maximum scale factor.
	 *
	 * @param[in] scaleFactor The new maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 */
	void setMaxScaleFactor(double scaleFactor) {
		maxScaleFactor = scaleFactor;
	}

	/**
	 * @return The size of the original image.
	 */
	cv::Size getImageSize() const;

private:

	size_t octaveLayerCount; ///< The number of layers per octave.
	double incrementalScaleFactor; ///< The incremental scale factor between two layers of the pyramid.
	double minScaleFactor; ///< The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	double maxScaleFactor; ///< The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	int firstLayer; ///< The index of the first stored pyramid layer.
	std::vector<std::shared_ptr<ImagePyramidLayer>> layers; ///< The pyramid layers.

	std::shared_ptr<VersionedImage> sourceImage; ///< The source image.
	std::shared_ptr<ImagePyramid> sourcePyramid; ///< The source pyramid.
	Version version; ///< The version.

	std::shared_ptr<ChainedFilter> imageFilter; ///< Filter that is applied to the image before down-scaling.
	std::shared_ptr<ChainedFilter> layerFilter; ///< Filter that is applied to the down-scaled images of the layers.
};

} /* namespace imageprocessing */
#endif // IMAGEPYRAMID_HPP_
