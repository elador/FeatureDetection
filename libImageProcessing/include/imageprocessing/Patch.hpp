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
		return x;
	}

	int getWidth() const {
		return data.cols;
	}

	int getHeight() const {
		return data.rows;
	}

	int getOriginalX() const {
		return cvRound(x / scale);
	}

	int getOriginalY() const {
		return cvRound(y / scale);
	}

	int getOriginalWidth() const {
		return cvRound(getWidth() / scale);
	}

	int getOriginalHeight() const {
		return cvRound(getHeight() / scale);
	}

	double getScale() const {
		return scale;
	}

	Mat getData() {
		return data;
	}

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
