/*
 * Patch.hpp
 *
 *  Created on: 15.02.2013
 *      Author: poschmann & huber
 */

#ifndef PATCH_HPP_
#define PATCH_HPP_

#include "opencv2/core/core.hpp"

namespace imageprocessing {

/**
 * Image patch with data, extracted from an image.
 */
class Patch {
public:

	/**
	 * Computes the bounds of a patch given by its center point and size.
	 *
	 * @param[in] center The center point of the patch.
	 * @param[in] size The size of the patch.
	 * @return The bounds of the patch.
	 */
	static cv::Rect computeBounds(cv::Point center, cv::Size size) {
		return cv::Rect(
				center.x - size.width / 2,
				center.y - size.height / 2,
				size.width,
				size.height
		);
	}

	/**
	 * Computes the center point of a patch given its bounds.
	 *
	 * @param[in] bounds The bounds of the patch.
	 * @return The center point of the patch.
	 */
	static cv::Point computeCenter(cv::Rect bounds) {
		return cv::Point(
				bounds.x + bounds.width / 2,
				bounds.y + bounds.height / 2
		);
	}

	/**
	 * Constructs a new empty patch. All values will be zero and the data will be empty.
	 */
	Patch() : center(0, 0), size(0, 0), data() {}

	/**
	 * Constructs a new patch.
	 *
	 * @param[in] x The original x-coordinate of the center of this patch.
	 * @param[in] y The original y-coordinate of the center of this patch.
	 * @param[in] width The original width.
	 * @param[in] height The original height.
	 * @param[in] data The patch data (might be an image patch or a feature vector).
	 */
	Patch(int x, int y, int width, int height, const cv::Mat& data) :
			center(x, y), size(width, height), data(data) {}

	/**
	 * Constructs a new patch given its bounding rectangle.
	 *
	 * @param[in] bounds The bounding rectangle of the patch.
	 * @param[in] data The patch data (might be an image patch or a feature vector).
	 */
	Patch(cv::Rect bounds, const cv::Mat& data) :
			center(computeCenter(bounds)), size(bounds.width, bounds.height), data(data) {}

	/**
	 * Copy constructor that clones the patch data.
	 *
	 * @param[in] other The patch that should be copied.
	 */
	Patch(const Patch& other) :
			center(other.center), size(other.size), data(other.data.clone()) {}

	/**
	 * Move constructor.
	 *
	 * @param[in] other The patch that should be moved.
	 */
	Patch(Patch&& other) :
			center(other.center), size(other.size), data(other.data) {
		other.data = cv::Mat();
	}

	/**
	 * Assignment operator that clones the patch data.
	 *
	 * @param[in] other The patch whose data should be assigned to this one.
	 */
	Patch& operator=(const Patch& other) {
		center = other.center;
		size = other.size;
		data = other.data.clone();
		return *this;
	}

	/**
	 * Move assignment operator.
	 *
	 * @param[in] other The patch whose data should be move assigned to this one.
	 */
	Patch& operator=(Patch&& other) {
		center = other.center;
		size = other.size;
		data = other.data;
		other.data = cv::Mat();
		return *this;
	}

	/**
	 * Determines whether this patch is equal to another one (without considering the data). Will compare the
	 * center coordinates (x and y) and size (width and height).
	 *
	 * @param[in] other The other patch.
	 * @return True if this patch is equal to the other one, false otherwise.
	 */
	bool operator==(const Patch& other) const {
		return center.x == other.center.x
				&& center.y == other.center.y
				&& size.width == other.size.width
				&& size.height == other.size.height;
	}

	/**
	 * @return The bounding rectangle of this patch.
	 */
	cv::Rect getBounds() const {
		return computeBounds(center, size);
	}

	/**
	 * @return The original x-coordinate of the center of this patch.
	 */
	int getX() const {
		return center.x;
	}

	/**
	 * @param[in] x The new original x-coordinate of the center of this patch.
	 */
	void setX(int x) {
		this->center.x = x;
	}

	/**
	 * @return The original y-coordinate of the center of this patch.
	 */
	int getY() const {
		return center.y;
	}

	/**
	 * @param[in] y The new original y-coordinate of the center of this patch.
	 */
	void setY(int y) {
		this->center.y = y;
	}

	/**
	 * @return The original width.
	 */
	int getWidth() const {
		return size.width;
	}

	/**
	 * @param[in] width The new original width.
	 */
	void setWidth(int width) {
		this->size.width = width;
	}

	/**
	 * @return The original height.
	 */
	int getHeight() const {
		return size.height;
	}

	/**
	 * @param[in] height The new original height.
	 */
	void setHeight(int height) {
		this->size.height = height;
	}

	/**
	 * @return The scale factor of the x-axis from the original size to the image data size.
	 */
	double getScaleFactorX() const {
		return static_cast<double>(data.cols) / static_cast<double>(size.width);
	}

	/**
	 * @return The scale factor of the y-axis from the original size to the image data size.
	 */
	double getScaleFactorY() const {
		return static_cast<double>(data.rows) / static_cast<double>(size.height);
	}

	/**
	 * @return The patch data (might be an image patch or a feature vector).
	 */
	cv::Mat& getData() {
		return data;
	}

	/**
	 * @return The patch data (might be an image patch or a feature vector).
	 */
	const cv::Mat& getData() const {
		return data;
	}

	/**
	 * Hash function for patches. Will only consider the center coordinates (x and y) and size (width and height).
	 */
	struct hash: std::unary_function<Patch, size_t> {
		size_t operator()(const Patch& patch) const {
			size_t prime = 31;
			size_t hash = 1;
			hash = prime * hash + std::hash<int>()(patch.center.x);
			hash = prime * hash + std::hash<int>()(patch.center.y);
			hash = prime * hash + std::hash<int>()(patch.size.width);
			hash = prime * hash + std::hash<int>()(patch.size.height);
			return hash;
		}
	};

private:

	cv::Point center; ///< The original center of this patch.
	cv::Size size; ///< The original size of this patch.
	cv::Mat data; ///< The patch data (might be an image patch or a feature vector).
};

} /* namespace imageprocessing */
#endif /* PATCH_HPP_ */
