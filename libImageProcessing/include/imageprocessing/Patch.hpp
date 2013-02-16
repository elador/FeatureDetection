/*
 * Patch.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann & huber
 */

#ifndef PATCH_HPP_
#define PATCH_HPP_

#include "opencv2/core/core.hpp"

using cv::Mat;

namespace imageprocessing {

/**
 * A possibly scaled image patch, extracted from an image pyramid.
 */
class Patch {
public:

	/**
	 * Constructs a new patch.
	 *
	 * @param[in] x The x-coordinate of the center of this patch inside its corresponding pyramid layer.
	 * @param[in] y The y-coordinate of the center of this patch inside its corresponding pyramid layer.
	 * @param[in] scale The scale factor of this patch in relation to its size in the original image.
	 * @param[in] data The image data.
	 */
	explicit Patch(int x, int y, double scale, Mat data);

	~Patch();

	/**
	 * @return The x-coordinate of the center of this patch inside its corresponding pyramid layer.
	 */
	int getX() const {
		return x;
	}

	/**
	 * @return The y-coordinate of the center of this patch inside its corresponding pyramid layer.
	 */
	int getY() const {
		return y;
	}

	/**
	 * @return The width of the data of this patch.
	 */
	int getWidth() const {
		return data.cols;
	}

	/**
	 * @return The height of the data of this patch.
	 */
	int getHeight() const {
		return data.rows;
	}

	/**
	 * @return The x-coordinate of the center of this patch in the original image.
	 */
	int getOriginalX() const {
		return cvRound(x / scale);
	}

	/**
	 * @return The y-coordinate of the center of this patch in the original image.
	 */
	int getOriginalY() const {
		return cvRound(y / scale);
	}

	/**
	 * @return The width of this patch in the original image.
	 */
	int getOriginalWidth() const {
		return cvRound(getWidth() / scale);
	}

	/**
	 * @return The height of this patch in the original image.
	 */
	int getOriginalHeight() const {
		return cvRound(getHeight() / scale);
	}

	/**
	 * @return The scale factor of this patch in relation to its size in the original image.
	 */
	double getScale() const {
		return scale;
	}

	/**
	 * @return The actual image data of this patch.
	 */
	Mat getData() {
		return data;
	}

	/**
	 * @return The actual image data of this patch.
	 */
	const Mat getData() const {
		return data;
	}

private:

	int x;        ///< The x-coordinate of the center of this patch inside its corresponding pyramid layer.
	int y;        ///< The y-coordinate of the center of this patch inside its corresponding pyramid layer.
	double scale; ///< The scale factor of this patch in relation to its size in the original image.
	Mat data;     ///< The image data.
};

} /* namespace imageprocessing */
#endif /* PATCH_HPP_ */
