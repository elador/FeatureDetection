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
#include <array>
#include <map>

using cv::Vec3f;
using cv::Size2f;
using std::string;
using std::array;
using std::map;

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
	 * @param[in] name The name of the landmark (TODO in tlms format).
	 * @param[in] position The 2D or 3D coordinates of the landmark.
	 * @param[in] size The width and height of the landmark, if it comes from an annotation or detector.
	 * @param[in] visibility A flag if the landmark is currently visible.
	 */
	explicit Landmark(string name, Vec3f position=Vec3f(0.0f, 0.0f, 0.0f), Size2f size=Size2f(0.0f, 0.0f), bool visibility=true);
	
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
	
	/**
	 * The width and height of the landmark, if it comes from an annotation or detector. Might be undefined (=0).
	 *
	 * @return The width and height of the landmark. Might be undefined (=0).
	 */
	Size2f getSize() const;


private:
	string name;		///< The name and identifier of the landmark.
	Vec3f position;		///< The 2D or 3D position of the landmark.
	Size2f size;		///< The width and height of the landmark, if it comes from an annotation or detector. Might be undefined (=0).
	bool visibility;	///< Flag if the landmark is currently visible.
	// maybe add a bool isRepresentedInModel later or something like that
};


/**
 * Represents a landmark in 2D or 3D. This can be a detected point
 * in an image, or vertex coordinates in a 3D model.
 */
class LandmarkSymbols {

public:
	static array<bool, 9> get(string landmarkName);
	static void getColor();

private:
	static map<string, array<bool, 9>> symbolMap;
}


} /* namespace imageio */
#endif /* LANDMARK_HPP_ */

/**
 * A note about the face-box: Maybe we want to handle the face-box completely separate from the landmarks.
 *       We hardly ever have the use-case, that we have a landmarks file/line with the face-box
 *       AND landmarks. We either have a tlms/did/... with only landmarks, or a lfw.lst with
 *       only the face-box information.
 *       Then, this Landmark class wouldn't need a size (width and height).
 *       But there's even more, maybe we some day want to use the ellipse head annotations from ALFW or some other database.
 *		 Probably, at that point, several classes are best, or/and maybe some struct with differend  landmark-types.
 */