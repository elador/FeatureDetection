/*
 * RectLandmark.hpp
 *
 *  Created on: 24.05.2013
 *      Author: poschmann
 */

#ifndef RECTLANDMARK_HPP_
#define RECTLANDMARK_HPP_

#include "imageio/Landmark.hpp"

using cv::Rect;
using cv::Point2f;

namespace imageio {

/**
 * Landmark that represents a rectangular area.
 */
class RectLandmark : public Landmark {
public:

	/**
	 * Constructs a new rectangular landmark based on another landmark.
	 *
	 * @param[in] landmark The landmark to take name, visibility, position and size from.
	 */
	explicit RectLandmark(const Landmark& landmark);

	/**
	 * Constructs a new invisible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 */
	explicit RectLandmark(const string& name);

	/**
	 * Constructs a new visible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] x The x coordinate of the center.
	 * @param[in] y The y coordinate of the center.
	 * @param[in] width The width.
	 * @param[in] height The height.
	 */
	RectLandmark(const string& name, float x, float y, float width, float height);

	/**
	 * Constructs a new visible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] rect The rectangular area.
	 */
	RectLandmark(const string& name, const Rect& rect);

	/**
	 * Constructs a new visible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] rect The rectangular area.
	 */
	RectLandmark(const string& name, const Rect_<float>& rect);

	/**
	 * Constructs a new visible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] position The position of the center.
	 * @param[in] size  The size.
	 */
	RectLandmark(const string& name, const Point2f& position, const Size2f& size);

	/**
	 * Constructs a new visible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] position The position of the center.
	 * @param[in] size The size.
	 */
	RectLandmark(const string& name, const Vec2f& position, const Size2f& size);

	/**
	 * Constructs a new rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] position The position of the center.
	 * @param[in] size The size.
	 * @param[in] visible A flag that indicates whether the landmark is visible.
	 */
	RectLandmark(const string& name, const Vec2f& position, const Size2f& size, bool visible);

	~RectLandmark();

	Vec2f getPosition2D() const {
		return position;
	}

	Vec3f getPosition3D() const {
		return Vec3f(getX(), getY(), 0);
	}

	Size2f getSize() const {
		return size;
	}

	float getX() const {
		return position[0];
	}

	float getY() const {
		return position[1];
	}

	float getZ() const {
		return 0;
	}

	float getWidth() const {
		return size.width;
	}

	float getHeight() const {
		return size.height;
	}

	bool isEqual(const Landmark& landmark) const;

	bool isClose(const Landmark& landmark, const float similarity) const;

	void draw(Mat& image, const Scalar& color = Scalar(0, 0, 0), float width = 1) const;

private:

	Vec2f position; ///< The position of the landmark.
	Size2f size;    ///< The size of the landmark.
};

} /* namespace imageio */
#endif /* RECTLANDMARK_HPP_ */
