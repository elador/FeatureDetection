/*
 * PyramidLayer.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#ifndef PYRAMIDLAYER_HPP_
#define PYRAMIDLAYER_HPP_

#include "opencv2/core/core.hpp"

using cv::Mat;

namespace imageprocessing {

/**
 * Layer of an image pyramid.
 */
class PyramidLayer {
public:

	/**
	 * Constructs a new pyramid layer.
	 *
	 * @param[in] scaleFactor The scale factor of this layer compared to the original image.
	 * @param[in] scaledImage The scaled image.
	 */
	explicit PyramidLayer(double scaleFactor, const Mat& scaledImage);

	virtual ~PyramidLayer();

	/**
	 * Computes the scaled representation of an original value (coordinate, size, ...) and rounds accordingly.
	 *
	 * @param[in] value The value in the original image.
	 * @return The corresponding value in this layer.
	 */
	int getScaled(int value) const {
		return cvRound(value / scaleFactor);
	}

	/**
	 * Computes the original representation of a scaled value (coordinate, size, ...) and rounds accordingly.
	 *
	 * @param[in] value The value in this layer.
	 * @return corresponding The value in the original image.
	 */
	int getOriginal(int value) const {
		return cvRound(value * scaleFactor);
	}

	/**
	 * @return The scale factor of this level compared to the original image.
	 */
	double getScaleFactor() const {
		return scaleFactor;
	}

	/**
	 * @return The scaled image.
	 */
	const Mat& getScaledImage() const {
		return scaledImage;
	}

private:

	double scaleFactor;    ///< The scale factor of this layer compared to the original image.
	const Mat scaledImage; ///< The scaled image.
};

} /* namespace imageprocessing */
#endif /* PYRAMIDLAYER_HPP_ */
