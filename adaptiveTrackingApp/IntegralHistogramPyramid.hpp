/*
 * ImagePyramid.hpp
 *
 *  Created on: 15.02.2013
 *      Author: huber & poschmann
 */

#pragma once

#ifndef INTEGRALHISTOGRAMPYRAMID_HPP_
#define INTEGRALHISTOGRAMPYRAMID_HPP_

#include "opencv2/core/core.hpp"
#include <vector>
#include <memory>
#include <array>

using cv::Mat;
using cv::Size;
using cv::Vec2b;
using std::vector;
using std::shared_ptr;
using std::array;

namespace imageprocessing {

class IntegralHistogramPyramidLayer;
class VersionedImage;
class ImagePyramid;
class ImageFilter;
class MultipleImageFilter;

/**
 * Image pyramid consisting of scaled representations of an image.
 */
class IntegralHistogramPyramid {
public:

	/**
	 * Constructs a new empty image pyramid.
	 *
	 * @param[in] bins The amount of bins to use.
	 * @param[in] signedGradients Flag that indicates whether signed gradients (direction from 0° to 360°) should be used.
	 * @param[in] offset Lower boundary of the first bin.
	 * @param[in] minScaleFactor The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 * @param[in] maxScaleFactor The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 * @param[in] incrementalScaleFactor The incremental scale factor between two layers of the pyramid.
	 */
	explicit IntegralHistogramPyramid(unsigned int bins, bool signedGradients = false, double offset = 0,
			double minScaleFactor = 0, double maxScaleFactor = 1, double incrementalScaleFactor = 0.9);

	/**
	 * Constructs a new empty image pyramid with another pyramid as its source.
	 *
	 * @param[in] bins The amount of bins to use.
	 * @param[in] signedGradients Flag that indicates whether signed gradients (direction from 0° to 360°) should be used.
	 * @param[in] offset Lower boundary of the first bin.
	 * @param[in] pyramid The new source pyramid.
	 * @param[in] minScaleFactor The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 * @param[in] maxScaleFactor The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 */
	explicit IntegralHistogramPyramid(unsigned int bins, bool signedGradients, double offset,
			shared_ptr<ImagePyramid> pyramid, double minScaleFactor = 0, double maxScaleFactor = 1);

	~IntegralHistogramPyramid();

	/**
	 * @return The version number.
	 */
	int getVersion() const {
		return version;
	}

	/**
	 * Changes the source to the given image. The pyramid will not get updated, therefore for the source change to take
	 * effect update() has to be called. The next call of update() will force an update based on the new image.
	 *
	 * @param[in] image The new source image.
	 */
	void setSource(const Mat& image);

	/**
	 * Changes the source to the given versioned image. The pyramid will not get updated, therefore for the source change
	 * to take effect update() has to be called. The next call of update() will only update this pyramid if the version of
	 * the given image differs from the one of this pyramid.
	 *
	 * @param[in] image The new source image.
	 */
	void setSource(shared_ptr<VersionedImage> image);

	/**
	 * Changes the source to the given pyramid. This pyramid will not get updated, therefore for the source change to take
	 * effect update() has to be called. The next call of update() will only update this pyramid if the version of the given
	 * pyramid differs from the one of this pyramid.
	 *
	 * @param[in] pyramid The new source pyramid.
	 */
	void setSource(shared_ptr<ImagePyramid> pyramid);

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
	void update(const Mat& image);

	/**
	 * Updates this pyramid using the saved parameters. If this pyramid's source is another pyramid, then that pyramid
	 * will be updated first with the given image. If this pyramid's source is an image or it has no source yet, the
	 * image will be the new source.
	 *
	 * @param[in] image The new image.
	 */
	void update(shared_ptr<VersionedImage> image);

	/**
	 * Determines the pyramid layer with the given index.
	 *
	 * @param[in] index The index of the pyramid layer.
	 * @return The pointer to a pyramid layer that may be empty if there is no layer with the given index.
	 */
	const shared_ptr<IntegralHistogramPyramidLayer> getLayer(int index) const;

	/**
	 * Determines the pyramid layer that is closest to the given scale factor.
	 *
	 * @param[in] scaleFactor The approximate scale factor of the layer.
	 * @return The pointer to a pyramid layer that may be empty if no layer has an appropriate scale factor.
	 */
	const shared_ptr<IntegralHistogramPyramidLayer> getLayer(double scaleFactor) const;

	/**
	 * Determines the size of the scaled image of each pyramid layer.
	 *
	 * @return The sizes of the pyramid layer's scaled images, beginning from the largest layer.
	 */
	vector<Size> getLayerSizes() const;

	/**
	 * @return A reference to the pyramid layers.
	 */
	const vector<shared_ptr<IntegralHistogramPyramidLayer>>& getLayers() const {
		return layers;
	}

	/**
	 * @return The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 */
	double getMinScaleFactor() const {
		return minScaleFactor;
	}

	/**
	 * @return The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 */
	double getMaxScaleFactor() const {
		return maxScaleFactor;
	}

	/**
	 * @return The incremental scale factor between two layers of the pyramid.
	 */
	double getIncrementalScaleFactor() const {
		return incrementalScaleFactor;
	}

	/**
	 * @return The size of the original image.
	 */
	Size getImageSize() const;

private:

	double minScaleFactor; ///< The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	double maxScaleFactor; ///< The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	double incrementalScaleFactor; ///< The incremental scale factor between two layers of the pyramid.
	int firstLayer;                               ///< The index of the first stored pyramid layer.
	vector<shared_ptr<IntegralHistogramPyramidLayer>> layers; ///< The pyramid layers.

	unsigned int bins; ///< The amount of bins.
	double offset; ///< Lower boundary of the first bin.
	array<Vec2b, 256 * 256> binCodes; ///< The look-up tables of the bin codes, the gradient codes are used as the index.

	shared_ptr<VersionedImage> sourceImage; ///< The source image.
	shared_ptr<ImagePyramid> sourcePyramid; ///< The source pyramid.
	int version;                            ///< The version number.

	shared_ptr<MultipleImageFilter> imageFilter; ///< Filter that is applied to the image before down-scaling.
	shared_ptr<MultipleImageFilter> layerFilter; ///< Filter that is applied to the down-scaled images of the layers.

	vector<Mat> createIntegralHistogram(const Mat& gradientImage);
};

} /* namespace imageprocessing */
#endif // INTEGRALHISTOGRAMPYRAMID_HPP_
