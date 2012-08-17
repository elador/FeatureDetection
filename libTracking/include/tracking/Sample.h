/*
 * Sample.h
 *
 *  Created on: 27.06.2012
 *      Author: poschmann
 */

#ifndef SAMPLE_H_
#define SAMPLE_H_

#include "tracking/Rectangle.h"

namespace tracking {

/**
 * Weighted sample representing an image region with three dimensions (x, y, size).
 */
class Sample {
public:

	/**
	 * Constructs a new sample.
	 */
	Sample();

	/**
	 * Constructs a new sample with a weight of one.
	 *
	 * @param[in] x The x coordinate.
	 * @param[in] y The y coordinate.
	 * @param[in] size The size (width).
	 */
	explicit Sample(int x, int y, int size);

	~Sample();

	/**
	 * @return The square bounding box representing the sample.
	 */
	inline Rectangle getBounds() const {
		int halfSize = size / 2;
		return Rectangle(x - halfSize, y - halfSize, size, size);
	}

	/**
	 * @return The x coordinate.
	 */
	inline int getX() const {
		return x;
	}

	/**
	 * Changes the x coordinate.
	 *
	 * @param[in] x The new x coordinate.
	 */
	inline void setX(int x) {
		this->x = x;
	}

	/**
	 * @return The y coordinate.
	 */
	inline int getY() const {
		return y;
	}

	/**
	 * Changes the y coordinate.
	 *
	 * @param[in] y The new y coordinate.
	 */
	inline void setY(int y) {
		this->y = y;
	}

	/**
	 * @return The size (width).
	 */
	inline int getSize() const {
		return size;
	}

	/**
	 * Changes the size (width).
	 *
	 * @param[in] size The new size.
	 */
	inline void setSize(int size) {
		this->size = size;
	}

	/**
	 * @return The weight.
	 */
	inline double getWeight() const {
		return weight;
	}

	/**
	 * Changes the weight.
	 *
	 * @param[in] weight The new weight.
	 */
	inline void setWeight(double weight) {
		this->weight = weight;
	}

	/**
	 * @return True if this sample represents the object, false otherwise.
	 */
	inline bool isObject() const {
		return object;
	}

	/**
	 * Changes whether this sample represents the object.
	 *
	 * @param[in] object Flag that indicates whether this sample represents the object.
	 */
	inline void setObject(bool object) {
		this->object = object;
	}

private:
	int x;         ///< The x coordinate.
	int y;         ///< The y coordinate.
	int size;      ///< The size (width).
	double weight; ///< The weight.
	bool object;   ///< Flag that indicates whether this sample represents the object.
};

} /* namespace tracking */
#endif /* SAMPLE_H_ */
