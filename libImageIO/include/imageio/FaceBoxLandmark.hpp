/*
 * FaceBoxLandmark.hpp
 *
 *  Created on: 23.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FACEBOXLANDMARK_HPP_
#define FACEBOXLANDMARK_HPP_

#include "imageio/Landmark.hpp"

namespace imageio {

/**
 * A landmark with a width and a height in addition, to represent a face-box.
 * Note: Maybe we want to handle the face-box completely separate from the landmarks. Let's see.
 *       We hardly ever have the use-case, that we have a landmarks file/line with the face-box
 *       AND landmarks. We either have a tlms/did/... with only landmarks, or a lfw.lst with
 *       only the face-box information.
 */
class FaceBoxLandmark : public Landmark {
public:

	/**
	 * Constructs a new Landmark of given name.
	 *
	 * @param[in] name The name of the landmark.
	 */
	FaceBoxLandmark(string name);
	
	FaceBoxLandmark(string name, Vec3f position, float width, float height, bool visibility=true);
	
	~FaceBoxLandmark();

	float getWidth() const;
	float getHeight() const;

private:
	float width;
	float height;

};

} /* namespace imageio */
#endif /* FACEBOXLANDMARK_HPP_ */
