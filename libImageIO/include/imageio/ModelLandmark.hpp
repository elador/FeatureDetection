/*
 * ModelLandmark.hpp
 *
 *  Created on: 22.03.2013
 *      Author: Patrik Huber
 */

#ifndef MODELLANDMARK_HPP_
#define MODELLANDMARK_HPP_

#include "imageio/Landmark.hpp"
#include <array>
#include <map>

using std::array;
using std::map;

namespace imageio {

/**
 * Landmark that represents a certain model point.
 */
class ModelLandmark : public Landmark {
public:

	/**
	 * Constructs a new invisible model landmark.
	 *
	 * @param[in] name The name of the landmark (TODO in tlms format).
	 */
	explicit ModelLandmark(const string& name);

	/**
	 * Constructs a new visible model landmark.
	 *
	 * @param[in] name The name of the landmark (TODO in tlms format).
	 * @param[in] x The x coordinate of the position.
	 * @param[in] y The y coordinate of the position.
	 * @param[in] z The z coordinate of the position.
	 */
	ModelLandmark(const string& name, float x, float y, float z = 0);

	/**
	 * Constructs a new visible model landmark.
	 *
	 * @param[in] name The name of the landmark (TODO in tlms format).
	 * @param[in] position The 2D coordinates of the landmark.
	 */
	ModelLandmark(const string& name, const Vec2f& position);

	/**
	 * Constructs a new visible model landmark.
	 *
	 * @param[in] name The name of the landmark (TODO in tlms format).
	 * @param[in] position The 3D coordinates of the landmark.
	 */
	ModelLandmark(const string& name, const Vec3f& position);

	/**
	 * Constructs a new model landmark.
	 *
	 * @param[in] name The name of the landmark (TODO in tlms format).
	 * @param[in] position The 2D or 3D coordinates of the landmark.
	 * @param[in] visible A flag that indicates whether the landmark is visible.
	 */
	ModelLandmark(const string& name, const Vec3f& position, bool visible);

	~ModelLandmark();

	Vec2f getPosition2D() const {
		return Vec2f(getX(), getY());
	}

	Vec3f getPosition3D() const {
		return position;
	}

	Size2f getSize() const {
		return Size2f(0, 0);
	}

	float getX() const {
		return position[0];
	}

	float getY() const {
		return position[1];
	}

	float getZ() const {
		return position[2];
	}

	float getWidth() const {
		return 0;
	}

	float getHeight() const {
		return 0;
	}

	bool isEqual(const Landmark& landmark) const;

	bool isClose(const Landmark& landmark, const float similarity) const;

	void draw(Mat& image, const Scalar& color = Scalar(0, 0, 0), float width = 1) const;

private:

	Vec3f position; ///< The 2D or 3D position of the landmark.
};


/**
 * Represents a landmark in 2D or 3D. This can be a detected point
 * in an image, or vertex coordinates in a 3D model.
 */
class LandmarkSymbols {

public:
	static array<bool, 9> get(string landmarkName);
	static cv::Scalar getColor(string landmarkName);

private:
	static map<string, array<bool, 9>> symbolMap;
	static map<string, cv::Scalar> colorMap;
};

} /* namespace imageio */
#endif /* MODELLANDMARK_HPP_ */
