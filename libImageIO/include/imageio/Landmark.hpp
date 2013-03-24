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

using cv::Vec3f;
using std::string;

namespace imageio {

/**
 * Represents a landmark in 2D or 3D. This can be a detected point
 * in an image, or vertex coordinates in a 3D model.
 */
class Landmark {
public:

	/**
	 * Constructs a new Landmark of given name.
	 *
	 * @param[in] name The name of the landmark.
	 * @param[in] position The 2D or 3D coordinates of the landmark.
	 * @param[in] visibility A flag if the landmark is currently visible.
	 */
	explicit Landmark(string name, Vec3f position=Vec3f(0.0f, 0.0f, 0.0f), bool visibility=true);
	
	~Landmark();

	/**
	 * Returns if the landmark is currently visible.
	 *
	 * @return True if the landmark is currently visible.
	 */
	bool isVisible() const;

	/**
	 * Returns the name of the landmark.
	 *
	 * @return The name of the landmark.
	 */
	string getName() const;

	/**
	 * Returns the 2D or 3D coordinates of the landmark.
	 *
	 * @return The 2D or 3D coordinates of the landmark.
	 */
	Vec3f getPosition() const;

private:
	string name;		///< The name and identifier of the landmark.
	Vec3f position;		///< The 2D or 3D position of the landmark.
	bool visibility;	///< Flag if the landmark is currently visible.
};

} /* namespace imageio */
#endif /* LANDMARK_HPP_ */
