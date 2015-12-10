/*
 * ImagePyramidLayer.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann
 */

#ifndef IMAGEPYRAMIDLAYER_HPP_
#define IMAGEPYRAMIDLAYER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include "opencv2/core/core.hpp"
#include <memory>

namespace imageprocessing {

/**
 * Layer of an image pyramid.
 */
class ImagePyramidLayer {
public:

	friend class ImagePyramid;

	/**
	 * Constructs a new pyramid layer.
	 *
	 * @param[in] index The index of this layer (0 is the original sized layer).
	 * @param[in] scale The (theoretical) scale factor of this layer compared to the original image.
	 * @param[in] scaleX The actual scale factor of the image width.
	 * @param[in] scaleY The actual scale factor of the image height.
	 * @param[in] image The image of this layer.
	 */
	ImagePyramidLayer(int index, double scale, double scaleX, double scaleY, const cv::Mat& image) :
			index(index), scale(scale), scaleX(scaleX), scaleY(scaleY), image(image) {}

	/**
	 * Constructs a copy of the given image pyramid layer that uses a different image.
	 *
	 * @param[in] other The layer to construct a copy of.
	 * @param[in] image The image of the copy.
	 */
	ImagePyramidLayer(const ImagePyramidLayer& other, const cv::Mat& image) :
			ImagePyramidLayer(other.index, other.scale, other.scale, other.scale, image) {}

	/**
	 * Creates a copy with a filtered image.
	 *
	 * @param[in] filter The filter to apply to the copied image.
	 * @return The copied image pyramid layer with a filtered image.
	 */
	std::shared_ptr<ImagePyramidLayer> createFiltered(const ImageFilter& filter) {
		return std::make_shared<ImagePyramidLayer>(*this, filter.applyTo(image));
	}

	/**
	 * Computes the scaled representation of an original value (coordinate, size, ...) and rounds accordingly.
	 *
	 * TODO bad, because a) it uses the theoretical scale factor and b) it rounds in a way that may be suboptimal
	 *      -> remove at some point, replace calls by getScaledX and getScaledY
	 *
	 * @param[in] value The value in the original image.
	 * @return The corresponding value in this layer.
	 */
	int getScaled(int value) const {
		return cvRound(value * scale);
	}

	/**
	 * Computes the scaled representation of an original x/column value (coordinate, size, ...).
	 *
	 * @param[in] x The x-value in the original image.
	 * @return The corresponding x-value in this layer.
	 */
	double getScaledX(double x) const {
		return x * scaleX;
	}

	/**
	 * Computes the scaled representation of an original y/row value (coordinate, size, ...).
	 *
	 * @param[in] y The y-value in the original image.
	 * @return The corresponding y-value in this layer.
	 */
	double getScaledY(double y) const {
		return y * scaleY;
	}

	/**
	 * Computes the original representation of a scaled value (coordinate, size, ...) and rounds accordingly.
	 *
	 * TODO bad, because a) it uses the theoretical scale factor and b) it rounds in a way that may be suboptimal
	 *      -> remove at some point, replace calls by getOriginalX and getOriginalY
	 *
	 * @param[in] value The value in this layer.
	 * @return corresponding The value in the original image.
	 */
	int getOriginal(int value) const {
		return cvRound(value / scale);
	}

	/**
	 * Computes the original representation of a scaled x/column value (coordinate, size, ...).
	 *
	 * @param[in] x The x-value in this layer.
	 * @return corresponding The x-value in the original image.
	 */
	double getOriginalX(double x) const {
		return x / scaleX;
	}

	/**
	 * Computes the original representation of a scaled y/row value (coordinate, size, ...).
	 *
	 * @param[in] y The y-value in this layer.
	 * @return corresponding The y-value in the original image.
	 */
	double getOriginalY(double y) const {
		return y / scaleY;
	}

	/**
	 * @return The size of this layer (size of the scaled image).
	 */
	cv::Size getSize() const {
		return cv::Size(image.cols, image.rows);
	}

	/**
	 * @return The index of this layer (0 is the original sized layer).
	 */
	int getIndex() const {
		return index;
	}

	/**
	 * @return The (theoretical) scale factor of this level compared to the original image.
	 */
	double getScaleFactor() const {
		return scale;
	}

	/**
	 * @return The image of this layer.
	 */
	const cv::Mat& getScaledImage() const {
		return image;
	}

	/**
	 * @return The image of this layer.
	 */
	cv::Mat& getScaledImage() {
		return image;
	}

private:

	int index; ///< The index of this layer (0 is the original sized layer).
	double scale; ///< The (theoretical) scale factor of this layer compared to the original image.
	double scaleX; ///< The actual scale factor of the image width.
	double scaleY; ///< The actual scale factor of the image height.
	cv::Mat image; ///< The image of this layer.
};

} /* namespace imageprocessing */
#endif /* IMAGEPYRAMIDLAYER_HPP_ */
