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

namespace imageio {

/**
 * Landmark that represents a certain model point.
 */
class ModelLandmark : public Landmark {
public:

	/**
	 * Constructs a new invisible model landmark.
	 *
	 * @param[in] name The name of the landmark.
	 */
	explicit ModelLandmark(const std::string& name);

	/**
	 * Constructs a new visible model landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] x The x coordinate of the position.
	 * @param[in] y The y coordinate of the position.
	 * @param[in] z The z coordinate of the position.
	 */
	ModelLandmark(const std::string& name, float x, float y, float z = 0);

	/**
	 * Constructs a new visible model landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] position The 2D coordinates of the landmark.
	 */
	ModelLandmark(const std::string& name, const cv::Vec2f& position);

	/**
	 * Constructs a new visible model landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] position The 3D coordinates of the landmark.
	 */
	ModelLandmark(const std::string& name, const cv::Vec3f& position);

	/**
	 * Constructs a new model landmark.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] position The 2D or 3D coordinates of the landmark.
	 * @param[in] visible A flag that indicates whether the landmark is visible.
	 */
	ModelLandmark(const std::string& name, const cv::Vec3f& position, bool visible);

	cv::Vec2f getPosition2D() const {
		return cv::Vec2f(getX(), getY());
	}

	cv::Vec3f getPosition3D() const {
		return position;
	}

	cv::Size2f getSize() const {
		return cv::Size2f(0, 0);
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

	// Todo doc: Compare two ModelLandmarks. If landmark is not model, returns...
	bool isEqual(const Landmark& landmark) const;

	// Todo doc: Compare two ModelLandmarks. If landmark is not model, returns...
	bool isClose(const Landmark& landmark, const float similarity) const;

	// Note: Expects a 3-channel image.
	// Color and width is not used
	void draw(cv::Mat& image, const cv::Scalar& color = cv::Scalar(0.0, 0.0, 0.0), float width = 1.0f) const;

private:

	cv::Vec3f position; ///< The 2D or 3D position of the landmark.
};


/**
 * Represents a landmark in 2D or 3D. This can be a detected point
 * in an image, or vertex coordinates in a 3D model.
 */
class LandmarkSymbols {

public:
	static std::array<bool, 9> get(std::string landmarkName);
	static cv::Scalar getColor(std::string landmarkName);

private:
	static std::map<std::string, std::array<bool, 9>> symbolMap;
	static std::map<std::string, cv::Scalar> colorMap;
};

} /* namespace imageio */
#endif /* MODELLANDMARK_HPP_ */
