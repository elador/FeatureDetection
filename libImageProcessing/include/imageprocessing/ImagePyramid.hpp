/*
 * ImagePyramid.hpp
 *
 *  Created on: 15.02.2013
 *      Author: huber & poschmann
 */

#pragma once

#ifndef IMAGEPYRAMID_H
#define IMAGEPYRAMID_H

#include "opencv2/core/core.hpp"
#include <vector>
#include <memory>

using cv::Mat;
using std::vector;
using std::shared_ptr;

namespace imageprocessing {

class ImagePyramidLayer;
class ImageFilter;
class MultipleImageFilter;

/**
 * Image pyramid consisting of scaled representations of an image.
 */
class ImagePyramid {
public:

	/**
	 * Constructs a new empty image pyramid.
	 *
	 * @param[in] minScaleFactor The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 * @param[in] maxScaleFactor The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 * @param[in] incrementalScaleFactor The incremental scale factor between two layers of the pyramid.
	 */
	ImagePyramid(double minScaleFactor = 0, double maxScaleFactor = 1, double incrementalScaleFactor = 0.9);

	/**
	 * Constructs a new empty image pyramid with another pyramid as its source.
	 *
	 * @param[in] pyramid The new source pyramid.
	 * @param[in] minScaleFactor The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 * @param[in] maxScaleFactor The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 */
	ImagePyramid(shared_ptr<ImagePyramid> pyramid, double minScaleFactor = 0, double maxScaleFactor = 1);

	~ImagePyramid();

	/**
	 * Changes the source to the given image. The pyramid will not get updated, therefore for the source change to take
	 * effect update() has to be called.
	 *
	 * @param[in] image The new source image.
	 */
	void setSource(const Mat& image);

	/**
	 * Changes the source to the given pyramid. This pyramid will not get updated, therefore for the source change to take
	 * effect update() has to be called.
	 *
	 * @param[in] pyramid The new source pyramid.
	 */
	void setSource(shared_ptr<ImagePyramid> pyramid);

	/**
	 * Forces an update of this pyramid based on the source (either an image or another pyramid).
	 */
	void update();

	/**
	 * Updates this pyramid using the saved parameters. If this pyramid's source is the same image as the one given, then
	 * the pyramid will remain unchanged even if the image content is different. If this pyramid's source is another
	 * pyramid, then that pyramid will be updated first with the given image. If this pyramid has no source yet, the image
	 * will be the new source.
	 *
	 * @param[in] image The new image.
	 */
	void update(const Mat& image);

	/**
	 * Adds a new image filter that is applied to the original image after the currently existing image filters.
	 *
	 * @param[in] filter The new image filter.
	 */
	void addImageFilter(shared_ptr<ImageFilter> filter);

	/**
	 * Adds a new image filter that is applied to the down-scaled images after the currently existing layer filters.
	 *
	 * @param[in] filter The new image filter.
	 */
	void addLayerFilter(shared_ptr<ImageFilter> filter);

	/**
	 * Determines the pyramid layer that is closest to the given scale factor.
	 *
	 * @param[in] scaleFactor The approximate scale factor of the layer.
	 * @return The pointer to a pyramid layer that may be empty if no layer has an appropriate scale factor.
	 */
	const shared_ptr<ImagePyramidLayer> getLayer(double scaleFactor) const;

	/**
	 * @return A reference to the pyramid layers.
	 */
	const vector<shared_ptr<ImagePyramidLayer>>& getLayers() const {
		return layers;
	}

private:

	double minScaleFactor; ///< The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	double maxScaleFactor; ///< The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	double incrementalScaleFactor; ///< The incremental scale factor between two layers of the pyramid.
	int firstLayer;                               ///< The index of the first stored pyramid layer.
	vector<shared_ptr<ImagePyramidLayer>> layers; ///< The pyramid layers.

	Mat sourceImage;                        ///< The source image.
	shared_ptr<ImagePyramid> sourcePyramid; ///< The source pyramid.

	shared_ptr<MultipleImageFilter> imageFilter; ///< Filter that is applied to the image before down-scaling.
	shared_ptr<MultipleImageFilter> layerFilter; ///< Filter that is applied to the down-scaled images of the layers.
};

} /* namespace imageprocessing */
#endif // IMAGEPYRAMID_H
