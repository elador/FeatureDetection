/*
 * MorphableModel.hpp
 *
 *  Created on: 30.09.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef MORPHABLEMODEL_HPP_
#define MORPHABLEMODEL_HPP_

#include "shapemodels/PcaModel.hpp"

#include "render/Mesh.hpp"

namespace shapemodels {

/**
 * Desc
 */
class MorphableModel  {
public:

	/**
	 * Constructs a new Morphable Model.
	 *
	 * @param[in] a b
	 */
	MorphableModel();
	
	/**
	 * Todo.
	 *
	 * @param[in] h5file Todo.
	 * @param[in] landmarkVertexMappingFile Todo.
	 * @return TODO.
	 */
	static MorphableModel loadOldBaselH5Model(std::string h5file, std::string landmarkVertexMappingFile);

	static MorphableModel loadScmModel(std::string h5file, std::string landmarkVertexMappingFile);
	
	PcaModel getShapeModel() const;
	PcaModel getColorModel() const;

	/**
	 * Returns the mean of the shape- and color model
	 * as a Mesh.
	 *
	 * @return The mean of the model.
	 */
	render::Mesh getMean() const;

	/**
	 * Return the value of the mean at a given landmark.
	 *
	 * @param[in] landmarkIdentifier A landmark identifier (e.g. "center.nose.tip").
	 * @return A Vec3f containing the values at the given landmark.
	 * @throws out_of_range exception if the landmarkIdentifier does not exist in the model. // TODO test the javadoc!
	 */
	//cv::Vec3f getMeanAtPoint(std::string landmarkIdentifier) const;

	/**
	 * Return the value of the mean at a given vertex id.
	 *
	 * @param[in] vertexIndex A vertex id.
	 * @return A Vec3f containing the values at the given vertex id.
	 */
	//cv::Vec3f getMeanAtPoint(unsigned int vertexIndex) const;

	/**
	 * Draws a random sample from the model, where the coefficients
	 * for the shape- and color models are both drawn from a standard
	 * normal (or with the given standard deviation).
	 *
	 * @param[in] sigma The standard deviation. (TODO find out which one, sigma=var, sigmaSq=sdev)
	 * @return A random sample from the model.
	 */
	render::Mesh drawSample(float sigma = 1.0f); // Todo sigmaShape, sigmaColor? or 2 functions?

	/**
	 * Returns a sample from the model with the given shape- and
	 * color PCA coefficients. 
	 * If a vector is empty, the mean is used.
	 *
	 * @param[in] shapeCoefficients The PCA coefficients used to generate the shape sample.
	 * @param[in] colorCoefficients The PCA coefficients used to generate the shape sample.
	 * @return A model instance with given coefficients.
	 */
	render::Mesh drawSample(std::vector<float> shapeCoefficients, std::vector<float> colorCoefficients);
	
private:
	PcaModel shapeModel; ///< A PCA model over the shape
	PcaModel colorModel; ///< A PCA model over vertex color information

};

} /* namespace shapemodels */
#endif /* MORPHABLEMODEL_HPP_ */
