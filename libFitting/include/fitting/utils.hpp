/*
 * utils.hpp
 *
 *  Created on: 04.12.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FITTINGUTILS_HPP_
#define FITTINGUTILS_HPP_

#include "morphablemodel/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

#include <vector>

namespace fitting {

/**
 * T.
 *
 * @param[in] a B.
 * @return c.
 */
class ModelFitting
{
public:
	ModelFitting()
	{

	};
	ModelFitting(float rx, float ry, float rz, float tx, float ty, boost::optional<float> tz, std::vector<float> shapeCoeffs, std::vector<float> albedoCoeffs, float focalLength) :
		rotationX(rx), rotationY(ry), rotationZ(rz), tx(tx), ty(ty), tz(tz), shapeCoeffs(shapeCoeffs), albedoCoeffs(albedoCoeffs), focalLength(focalLength)
	{

	};
	// Note: Better would be to internally store the rotation as a quaternion 'rotation'
	// Model orientation:
	float rotationX;
	float rotationY;
	float rotationZ;

	float tx;
	float ty;
	boost::optional<float> tz;
	
	std::vector<float> shapeCoeffs;
	std::vector<float> albedoCoeffs; // Rename to ModelState and move to libMorphableModel?
	
	// Camera parameters:
	float focalLength;

	// We need this too for the window transform / fov calculation?
	// Maybe separate stuff into "ModelInstance" or "ModelFitting", and "ImageFittingResult" (bad name...) or something?
	// int imageWidth, imageHeight

	// AngleConvention (not needed when stored as quaternion?)
	// ProjectionType?
	// Texture information? Model information?

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & rotationX;
		ar & rotationY;
		ar & rotationZ;
		ar & tx;
		ar & ty;
		ar & tz;
		ar & shapeCoeffs;
		ar & albedoCoeffs;
	};

};

} /* namespace fitting */
#endif /* FITTINGUTILS_HPP_ */
