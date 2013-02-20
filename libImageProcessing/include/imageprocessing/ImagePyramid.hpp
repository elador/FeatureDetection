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
	 */
	ImagePyramid();

	~ImagePyramid();

	/**
	 * Re-builds this pyramid based on an image.
	 *
	 * @param[in] image The new source image.
	 * @param[in] incrementalScaleFactor The incremental scale factor between two layers of the pyramid.
	 * @param[in] maxScaleFactor The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 * @param[in] minScaleFactor The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 */
	void build(const Mat& image, double incrementalScaleFactor, double maxScaleFactor, double minScaleFactor);

	/**
	 * Re-builds this pyramid based on another pyramid.
	 *
	 * @param[in] pyramid The new source pyramid.
	 */
	void build(shared_ptr<ImagePyramid> pyramid);

	/**
	 * Re-builds this pyramid based on another pyramid with additional scale restrictions.
	 *
	 * @param[in] pyramid The new source pyramid.
	 * @param[in] maxScaleFactor The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	 * @param[in] minScaleFactor The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	 */
	void build(shared_ptr<ImagePyramid> pyramid, double maxScaleFactor, double minScaleFactor);

	/**
	 * Updates this pyramid using the saved parameters. If this pyramid was updated or built with the
	 * same image, then the pyramid will remain unchanged even if the image content is different. If
	 * this pyramid was built using another pyramid, then that pyramid will be updated first.
	 *
	 * @param[in] image The new image.
	 */
	void update(const Mat& image);

	/**
	 * Adds a new image filter that is applied to original image after the currently existing image filters.
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

	double incrementalScaleFactor; ///< The incremental scale factor between two layers of the pyramid.
	double maxScaleFactor; ///< The maximum scale factor (the scale factor of the biggest scaled (first) image is less or equal).
	double minScaleFactor; ///< The minimum scale factor (the scale factor of the smallest scaled (last) image is bigger or equal).
	int firstLayer;                               ///< The index of the first stored pyramid layer.
	vector<shared_ptr<ImagePyramidLayer>> layers; ///< The pyramid layers.

	Mat sourceImage;                        ///< The source image.
	shared_ptr<ImagePyramid> sourcePyramid; ///< The source pyramid.

	shared_ptr<MultipleImageFilter> imageFilter; ///< Filter that is applied to the image before down-scaling.
	shared_ptr<MultipleImageFilter> layerFilter; ///< Filter that is applied to the down-scaled images of the layers.
};

} /* namespace imageprocessing */
#endif // IMAGEPYRAMID_H
