/*
 * Sample.h
 *
 *  Created on: 27.06.2012
 *      Author: poschmann
 */

#ifndef SAMPLE_H_
#define SAMPLE_H_

#include "condensation/Rectangle.h"

namespace condensation {

/**
 * Weighted sample representing an image region with three dimensions (x, y, size).
 */
class Sample {
public:

	/**
	 * Constructs a new sample.
	 */
	Sample() : x(0), y(0), size(0), weight(1), object(false) {}

	/**
	 * Constructs a new sample with a weight of one.
	 *
	 * @param[in] x The x coordinate.
	 * @param[in] y The y coordinate.
	 * @param[in] size The size (width).
	 */
	Sample(int x, int y, int size) : x(x), y(y), size(size), weight(0), object(false) {}

	~Sample() {}

	/**
	 * @return The square bounding box representing the sample.
	 */
	Rectangle getBounds() const {
		int halfSize = size / 2;
		return Rectangle(x - halfSize, y - halfSize, size, size);
	}

	/**
	 * @return The x coordinate.
	 */
	int getX() const {
		return x;
	}

	/**
	 * Changes the x coordinate.
	 *
	 * @param[in] x The new x coordinate.
	 */
	void setX(int x) {
		this->x = x;
	}

	/**
	 * @return The y coordinate.
	 */
	int getY() const {
		return y;
	}

	/**
	 * Changes the y coordinate.
	 *
	 * @param[in] y The new y coordinate.
	 */
	void setY(int y) {
		this->y = y;
	}

	/**
	 * @return The size (width).
	 */
	int getSize() const {
		return size;
	}

	/**
	 * Changes the size (width).
	 *
	 * @param[in] size The new size.
	 */
	void setSize(int size) {
		this->size = size;
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
	int x;         ///< The x coordinate.
	int y;         ///< The y coordinate.
	int size;      ///< The size (width).
	double weight; ///< The weight.
	bool object;   ///< Flag that indicates whether this sample represents the object.
};

} /* namespace condensation */
#endif /* SAMPLE_H_ */
