/*
 * Sample.hpp
 *
 *  Created on: 27.06.2012
 *      Author: poschmann
 */

#ifndef SAMPLE_HPP_
#define SAMPLE_HPP_

#include "opencv2/core/core.hpp"

using cv::Rect;

namespace condensation {

/**
 * Weighted sample representing a square image region with position and size (x, y, size) and according change (vx, vy, vsize).
 */
class Sample {
public:

	/**
	 * Constructs a new sample.
	 */
	Sample() : x(0), y(0), size(0), vx(0), vy(0), vsize(0), weight(1), object(false) {}

	/**
	 * Constructs a new sample with velocities of zero and a weight of one.
	 *
	 * @param[in] x The x coordinate of the center.
	 * @param[in] y The y coordinate of the center.
	 * @param[in] size The size.
	 */
	Sample(int x, int y, int size) : x(x), y(y), size(size), vx(0), vy(0), vsize(0),  weight(1), object(false) {}

	/**
	 * Constructs a new sample with a weight of one.
	 *
	 * @param[in] x The x coordinate of the center.
	 * @param[in] y The y coordinate of the center.
	 * @param[in] size The size.
	 * @param[in] vx The change of the x coordinate.
	 * @param[in] vy The change of the y coordinate.
	 * @param[in] vsize The change of the size.
	 */
	Sample(int x, int y, int size, int vx, int vy, int vsize) :
			x(x), y(y), size(size), vx(vx), vy(vy), vsize(vsize),  weight(1), object(false) {}

	~Sample() {}

	/**
	 * @return The square bounding box representing this sample.
	 */
	Rect getBounds() const {
		int halfSize = size / 2;
		return Rect(x - halfSize, y - halfSize, size, size);
	}

	/**
	 * @return The x coordinate of the center.
	 */
	int getX() const {
		return x;
	}

	/**
	 * @param[in] x The new x coordinate of the center.
	 */
	void setX(int x) {
		this->x = x;
	}

	/**
	 * @return The y coordinate of the center.
	 */
	int getY() const {
		return y;
	}

	/**
	 * @param[in] y The new y coordinate of the center.
	 */
	void setY(int y) {
		this->y = y;
	}

	/**
	 * @return The size.
	 */
	int getSize() const {
		return size;
	}

	/**
	 * @param[in] size The new size.
	 */
	void setSize(int size) {
		this->size = size;
	}

	/**
	 * @return The change of the x coordinate.
	 */
	int getVx() const {
		return vx;
	}

	/**
	 * @param[in] x The new change of the x coordinate.
	 */
	void setVx(int vx) {
		this->vx = vx;
	}

	/**
	 * @return The change of the y coordinate.
	 */
	int getVy() const {
		return vy;
	}

	/**
	 * @param[in] y The new change of the y coordinate.
	 */
	void setVy(int vy) {
		this->vy = vy;
	}

	/**
	 * @return The change of the size.
	 */
	int getVSize() const {
		return vsize;
	}

	/**
	 * @param[in] size The new change of the size.
	 */
	void setVSize(int vsize) {
		this->vsize = vsize;
	}

	/**
	 * @return The weight.
	 */
	double getWeight() const {
		return weight;
	}

	/**
	 * Changes the weight.
	 *
	 * @param[in] weight The new weight.
	 */
	void setWeight(double weight) {
		this->weight = weight;
	}

	/**
	 * @return True if this sample represents the object, false otherwise.
	 */
	bool isObject() const {
		return object;
	}

	/**
	 * Changes whether this sample represents the object.
	 *
	 * @param[in] object Flag that indicates whether this sample represents the object.
	 */
	void setObject(bool object) {
		this->object = object;
	}

	/**
	 * Determines whether this sample is less than another sample using the weight. This sample is considered
	 * less than the other sample if the weight of this one is less than the weight of the other sample.
	 *
	 * @param[in] other The other sample.
	 * @return True if this sample comes before the other in a strict weak ordering, false otherwise.
	 */
	bool operator<(const Sample& other) const {
		return weight < other.weight;
	}

	/**
	 * Determines whether this sample is bigger than another sample using the weight. This sample is considered
	 * bigger than the other sample if the weight of this one is bigger than the weight of the other sample.
	 *
	 * @param[in] other The other sample.
	 * @return True if this sample comes before the other in a strict weak ordering, false otherwise.
	 */
	bool operator>(const Sample& other) const {
		return weight > other.weight;
	}

	/**
	 * Comparison function that compares samples by their weight in ascending order.
	 */
	class WeightComparisonAsc {
	public:
		bool operator()(const Sample& lhs, const Sample& rhs) {
			return lhs.weight < rhs.weight;
		}
	};

	/**
	 * Comparison function that compares samples by their weight in descending order.
	 */
	class WeightComparisonDesc {
	public:
		bool operator()(const Sample& lhs, const Sample& rhs) {
			return lhs.weight > rhs.weight;
		}
	};

private:
	int x;         ///< The x coordinate of the center.
	int y;         ///< The y coordinate of the center.
	int size;      ///< The size.
	int vx;        ///< The change of the x coordinate.
	int vy;        ///< The change of the y coordinate.
	int vsize;     ///< The change of the size.
	double weight; ///< The weight.
	bool object;   ///< Flag that indicates whether this sample represents the object.
};

} /* namespace condensation */
#endif /* SAMPLE_HPP_ */
