/*
 * RectLandmark.hpp
 *
 *  Created on: 24.05.2013
 *      Author: poschmann
 */
#pragma once

#ifndef RECTLANDMARK_HPP_
#define RECTLANDMARK_HPP_

#include "imageio/Landmark.hpp"

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
	explicit RectLandmark(const std::string& name);

	/**
	 * Constructs a new visible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] x The x coordinate of the center.
	 * @param[in] y The y coordinate of the center.
	 * @param[in] width The width.
	 * @param[in] height The height.
	 */
	RectLandmark(const std::string& name, float x, float y, float width, float height);

	/**
	 * Constructs a new visible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] rect The rectangular area.
	 */
	RectLandmark(const std::string& name, const cv::Rect& rect);

	/**
	 * Constructs a new visible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] rect The rectangular area.
	 */
	RectLandmark(const std::string& name, const cv::Rect_<float>& rect);

	/**
	 * Constructs a new visible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] position The position of the center.
	 * @param[in] size  The size.
	 */
	RectLandmark(const std::string& name, const cv::Point2f& position, const cv::Size2f& size);

	/**
	 * Constructs a new visible rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] position The position of the center.
	 * @param[in] size The size.
	 */
	RectLandmark(const std::string& name, const cv::Vec2f& position, const cv::Size2f& size);

	/**
	 * Constructs a new rectangular landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] position The position of the center.
	 * @param[in] size The size.
	 * @param[in] visible A flag that indicates whether the landmark is visible.
	 */
	RectLandmark(const std::string& name, const cv::Vec2f& position, const cv::Size2f& size, bool visible);

	cv::Vec2f getPosition2D() const {
		return position;
	}

	cv::Vec3f getPosition3D() const {
		return cv::Vec3f(getX(), getY(), 0);
	}

	cv::Size2f getSize() const {
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

	void draw(cv::Mat& image, const cv::Scalar& color = cv::Scalar(0.0, 0.0, 255.0), float width = 1.0f) const;

private:

	cv::Vec2f position; ///< The position of the landmark.
	cv::Size2f size;    ///< The size of the landmark.
};

} /* namespace imageio */
#endif /* RECTLANDMARK_HPP_ */
