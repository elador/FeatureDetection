/*
 * Landmark.hpp
 *
 *  Created on: 22.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LANDMARK_HPP_
#define LANDMARK_HPP_

#include "opencv2/core/core.hpp"
#include <string>

namespace imageio {

/**
 * Represents some kind of landmark. This can be a detected point in an image,
 * a vertex in a 3D model or some kind of shape describing an object position
 * within an image.
 */
class Landmark {
public:

	enum class LandmarkType { MODEL, RECT };

	/**
	 * Constructs a new landmark.
	 *
	 * @param[in] type The type of the landmark.
	 * @param[in] name The name of the landmark.
	 * @param[in] visible A flag that indicates whether the landmark is visible.
	 */
	Landmark(LandmarkType type, const std::string& name, bool visible) : type(type), name(name), visible(visible) {}

	virtual ~Landmark() {}

	/**
	 * @return The type of the landmark.
	 */
	const LandmarkType& getType() const {
		return type;
	}

	/**
	 * @return The name of the landmark.
	 */
	const std::string& getName() const {
		return name;
	}

	/**
	 * @return True if the landmark is visible, false otherwise.
	 */
	bool isVisible() const {
		return visible;
	}

	/**
	 * @return The 2D coordinates of the landmark's center.
	 */
	cv::Point2f getPoint2D() const {
		return cv::Point2f(getX(), getY());
	}

	/**
	 * @return The 3D coordinates of the landmark's center.
	 */
	cv::Point3f getPoint3D() const {
		return cv::Point3f(getX(), getY(), getZ());
	}

	/**
	 * @return The rectangular area of this landmark. In case of a landmark with no size, the size of the rect will be zero.
	 */
	cv::Rect_<float> getRect() const {
		return cv::Rect_<float>(getX() - 0.5f * getWidth(), getY() - 0.5f * getHeight(), getWidth(), getHeight());
	}

	/**
	 * @return The 2D coordinates of the landmark's center.
	 */
	virtual cv::Vec2f getPosition2D() const = 0;

	/**
	 * @return The 3D coordinates of the landmark's center.
	 */
	virtual cv::Vec3f getPosition3D() const = 0;
	
	/**
	 * The width and height of the landmark, if it comes from an annotation or detector. Might be undefined (=0).
	 *
	 * @return The width and height of the landmark. Might be undefined (=0).
	 */
	virtual cv::Size2f getSize() const = 0;

	/**
	 * @return The x coordinate of the center.
	 */
	virtual float getX() const = 0;

	/**
	 * @return The y coordinate of the center.
	 */
	virtual float getY() const = 0;

	/**
	 * @return The z coordinate of the center.
	 */
	virtual float getZ() const = 0;

	/**
	 * @return The width.
	 */
	virtual float getWidth() const = 0;

	/**
	 * @return The height.
	 */
	virtual float getHeight() const = 0;

	/**
	 * Compares this landmark against another one and returns true if they
	 * are within a small epsilon value.
	 *
	 * Note: |x-y|<=eps doesn't seem to be something good to do according to google but
	 *       it should suffice for our purpose.
	 *
	 * @param[in] landmark A landmark to compare this landmark to.
	 * @return True if the two landmarks are equal, false otherwise.
	 */
	virtual bool isEqual(const Landmark& landmark) const = 0;

	/**
	 * Compares this landmark against another one and returns true if they are
	 * close to each other. 
	 * If the landmark possesses only a position, and no size, the second parameter
	 * specifies an absolute distance to compare against.
	 * If the landmark possesses a size (width/height), the second parameter
	 * specifies a ratio (between 0 and 1) by which the two landmarks must overlap
	 * to still be considered close. Overlap is the ratio between intersection and
	 * union of the areas of the landmarks.
	 *
	 * @param[in] landmark A landmark to compare this landmark to.
	 * @param[in] similarity A value in pixels or a ratio, depending if the
	 *            landmark has a size.
	 * @return True if the two landmarks are close, false otherwise.
	 */
	virtual bool isClose(const Landmark& landmark, const float similarity) const = 0;

	/**
	 * Draws this landmark at its position in the image. Its corresponding
	 * symbol is used if found in the LandmarkSymbols map - if not, a default
	 * symbol is drawn.
	 *
	 * @param[in] image The image into which this landmark is drawn.
	 */
	virtual void draw(cv::Mat& image, const cv::Scalar& color = cv::Scalar(0.0, 0.0, 255.0), float width = 1.0f) const = 0;

private:

	LandmarkType type; ///< The type of the landmark.
	std::string name;       ///< The name and identifier of the landmark.
	bool visible;      ///< Flag that indicates whether the landmark is visible.
};

} /* namespace imageio */
#endif /* LANDMARK_HPP_ */
