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
	 * Constructs a new empty patch. All values will be zero and the data will be empty.
	 */
	Patch() :
			x(0), y(0), width(0), height(0), data() {}

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
			x(x), y(y), width(width), height(height), data(data) {}

	/**
	 * Constructs a new patch given its bounding rectangle.
	 *
	 * @param[in] bounds The bounding rectangle of the patch.
	 * @param[in] data The patch data (might be an image patch or a feature vector).
	 */
	Patch(cv::Rect bounds, const cv::Mat& data) :
		  x(bounds.x + bounds.width / 2), y(bounds.y + bounds.height / 2), width(bounds.width), height(bounds.height) {}

	/**
	 * Copy constructor that clones the patch data.
	 *
	 * @param[in] other The patch that should be copied.
	 */
	Patch(const Patch& other) :
			x(other.x), y(other.y), width(other.width), height(other.height), data(other.data.clone()) {}

	/**
	 * Move constructor.
	 *
	 * @param[in] other The patch that should be moved.
	 */
	Patch(Patch&& other) :
			x(other.x), y(other.y), width(other.width), height(other.height), data(other.data) {}

	~Patch() {}

	/**
	 * Assignment operator that clones the patch data.
	 *
	 * @param[in] other The patch whose data should be assigned to this one.
	 */
	Patch& operator=(const Patch& other) {
		x = other.x;
		y = other.y;
		width = other.width;
		height = other.height;
		data = other.data.clone();
		return *this;
	}

	/**
	 * Move assignment operator.
	 *
	 * @param[in] other The patch whose data should be move assigned to this one.
	 */
	Patch& operator=(const Patch&& other) {
		x = other.x;
		y = other.y;
		width = other.width;
		height = other.height;
		data = other.data;
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
		return x == other.x && y == other.y && width == other.width && height == other.height;
	}

	/**
	 * @return The bounding rectangle of this patch.
	 */
	cv::Rect getBounds() const {
		return cv::Rect(x - width / 2, y - height / 2, width, height);
	}

	/**
	 * @return The original x-coordinate of the center of this patch.
	 */
	int getX() const {
		return x;
	}

	/**
	 * @param[in] x The new original x-coordinate of the center of this patch.
	 */
	void setX(int x) {
		this->x = x;
	}

	/**
	 * @return The original y-coordinate of the center of this patch.
	 */
	int getY() const {
		return y;
	}

	/**
	 * @param[in] y The new original y-coordinate of the center of this patch.
	 */
	void setY(int y) {
		this->y = y;
	}

	/**
	 * @return The original width.
	 */
	int getWidth() const {
		return width;
	}

	/**
	 * @param[in] width The new original width.
	 */
	void setWidth(int width) {
		this->width = width;
	}

	/**
	 * @return The original height.
	 */
	int getHeight() const {
		return height;
	}

	/**
	 * @param[in] height The new original height.
	 */
	void setHeight(int height) {
		this->height = height;
	}

	/**
	 * @return The scale factor of the x-axis from the original size to the image data size.
	 */
	double getScaleFactorX() const {
		return static_cast<double>(data.cols) / static_cast<double>(width);
	}

	/**
	 * @return The scale factor of the y-axis from the original size to the image data size.
	 */
	double getScaleFactorY() const {
		return static_cast<double>(data.rows) / static_cast<double>(height);
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
			hash = prime * hash + std::hash<int>()(patch.x);
			hash = prime * hash + std::hash<int>()(patch.y);
			hash = prime * hash + std::hash<int>()(patch.width);
			hash = prime * hash + std::hash<int>()(patch.height);
			return hash;
		}
	};

private:

	int x;        ///< The original x-coordinate of the center of this patch.
	int y;        ///< The original y-coordinate of the center of this patch.
	int width;    ///< The original width.
	int height;   ///< The original height.
	cv::Mat data; ///< The patch data (might be an image patch or a feature vector).
};

} /* namespace imageprocessing */
#endif /* PATCH_HPP_ */
