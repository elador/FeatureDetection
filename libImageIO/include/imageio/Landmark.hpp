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
using cv::Mat;
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
	bool isEqual(const Landmark& landmark) const;

	/**
	 * Compares this landmark against another one and returns true if they are
	 * close to each other. 
	 * If the landmark possesses only a position, and no size, the second parameter
	 * specifies an absolute distance to compare against.
	 * If the landmark possesses a size (width/height), the second parameter
	 * specifies a ratio (between 0 and 1) by which the two landmarks must overlap
	 * to still be considered close.
	 *
	 * @param[in] landmark A landmark to compare this landmark to.
	 * @param[in] similarity A value in pixels or a ratio, depending if the
	 *            landmark has a size.
	 * @return True if the two landmarks are close, false otherwise.
	 */
	bool isClose(const Landmark& landmark, const float similarity) const;

	/**
	 * Draws this landmark at its position in the image. Its corresponding
	 * symbol is used if found in the LandmarkSymbols map - if not, a default
	 * symbol is drawn.
	 *
	 * @param[in] image The image into which this landmark is drawn.
	 */
	void draw(Mat image) const;

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
	static cv::Scalar getColor(string landmarkName);

private:
	static map<string, array<bool, 9>> symbolMap;
	static map<string, cv::Scalar> colorMap;
};


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