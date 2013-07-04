/*
 * ImagePyramidLayer.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#ifndef INTEGRALHISTOGRAMPYRAMIDLAYER_HPP_
#define INTEGRALHISTOGRAMPYRAMIDLAYER_HPP_

#include "opencv2/core/core.hpp"
#include <vector>

using cv::Mat;
using cv::Size;
using std::vector;

namespace imageprocessing {

/**
 * Layer of an image pyramid.
 */
class IntegralHistogramPyramidLayer {
public:

	/**
	 * Constructs a new pyramid layer.
	 *
	 * @param[in] index The index of this layer (0 is the original sized layer).
	 * @param[in] scaleFactor The scale factor of this layer compared to the original image.
	 * @param[in] scaledImage The scaled integral histogram.
	 */
	explicit IntegralHistogramPyramidLayer(int index, double scaleFactor, const vector<Mat>& scaledImage);

	~IntegralHistogramPyramidLayer();

	/**
	 * Computes the scaled representation of an original value (coordinate, size, ...) and rounds accordingly.
	 *
	 * @param[in] value The value in the original image.
	 * @return The corresponding value in this layer.
	 */
	int getScaled(int value) const {
		return cvRound(value * scaleFactor);
	}

	/**
	 * Computes the original representation of a scaled value (coordinate, size, ...) and rounds accordingly.
	 *
	 * @param[in] value The value in this layer.
	 * @return corresponding The value in the original image.
	 */
	int getOriginal(int value) const {
		return cvRound(value / scaleFactor);
	}

	/**
	 * @return The size of this layer (size of the scaled image).
	 */
	Size getSize() const {
		return Size(scaledImage[0].cols, scaledImage[0].rows);
	}

	/**
	 * @return The index of this layer (0 is the original sized layer).
	 */
	int getIndex() const {
		return index;
	}

	/**
	 * @return The scale factor of this level compared to the original image.
	 */
	double getScaleFactor() const {
		return scaleFactor;
	}

	/**
	 * @return The scaled integral histogram.
	 */
	const vector<Mat>& getScaledImage() const {
		return scaledImage;
	}

	/**
	 * @return The scaled integral histogram.
	 */
	vector<Mat>& getScaledImage() {
		return scaledImage;
	}

private:

	int index;          ///< The index of this layer (0 is the original sized layer).
	double scaleFactor; ///< The scale factor of this layer compared to the original image.
	vector<Mat> scaledImage;    ///< The scaled integral histogram.
};

} /* namespace imageprocessing */
#endif /* INTEGRALHISTOGRAMPYRAMIDLAYER_HPP_ */
