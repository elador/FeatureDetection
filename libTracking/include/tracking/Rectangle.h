/*
 * Rectangle.h
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#ifndef RECTANGLE_H_
#define RECTANGLE_H_

namespace tracking {

/**
 * Rectangle in image coordinates.
 */
class Rectangle {
public:

	/**
	 * Constructs a new rectangle.
	 *
	 * @param[in] x The x coordinate of the upper left corner.
	 * @param[in] y The y coordinate of the upper left corner.
	 * @param[in] w The width.
	 * @param[in] h The height.
	 */
	explicit Rectangle(int x, int y, int w, int h);
	~Rectangle();

	/**
	 * @return The x coordinate of the upper left corner.
	 */
	inline int getX() const {
		return x;
	}

	/**
	 * @return The y coordinate of the upper left corner.
	 */
	inline int getY() const {
		return y;
	}

	/**
	 * @return The width.
	 */
	inline int getWidth() const {
		return w;
	}

	/**
	 * @return The height.
	 */
	inline int getHeight() const {
		return w;
	}

private:
	const int x;	///< The x coordinate of the upper left corner.
	const int y;	///< The y coordinate of the upper left corner.
	const int w;	///< The width.
	const int h;	///< The height.
};

} /* namespace tracking */
#endif /* RECTANGLE_H_ */
