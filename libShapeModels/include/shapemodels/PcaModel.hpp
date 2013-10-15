/*
 * PcaModel.hpp
 *
 *  Created on: 30.09.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef PCAMODEL_HPP_
#define PCAMODEL_HPP_

#include "opencv2/core/core.hpp"

#include <string>
#include <vector>
#include <array>
#include <map>
#include <random>

namespace shapemodels {

/**
 * A PCA-model that consists of:
 *   - a mean vector (y x z)
 *   - a PCA basis matrix
 *   - a PCA variance vector
 *   - optionally a scalar with the noise variance.
 *
 * It also contains a list of triangles to built a mesh as well as a mapping
 * from landmark points to the corresponding vertex-id in the mesh.
 * It is able to return instances of the model as meshes.
 */
class PcaModel {
public:
	
	/**
	 * Specifies the type of the PCA model.
	 * Mainly used so that the loading function knows which part of the data to read.
	 */
	enum class ModelType {
		SHAPE,  ///< A model where the data corresponds to shape information (vertex positions)
		COLOR ///< A model where the data corresponds to color information (vertex-coloring)
	};

	/**
	 * Constructs an empty PCA model.
	 * It is recommended to use one of the static load methods instead.
	 */
	PcaModel();

	/**
	 * Load a shape or color model from a .scm file containing
	 * a Morphable Model in the Surrey format.
	 *
	 * @param[in] modelFile A binary .scm-file containing the model.
	 * @param[in] landmarkVertexMappingFile A file containing a mapping from landmarks to vertex ids.
	 * @param[in] modelType The type of PCA model to load (SHAPE or COLOR).
	 * @return A shape- or color model from the given file.
	 */
	static PcaModel loadScmModel(std::string modelFilename, std::string landmarkVertexMappingFile, ModelType modelType);

	/**
	 * Load a shape or color model from a .h5 file containing a Morphable Model 
	 * from Basel. The format is deprecated, it's more or less the one that
	 * was used during my MSc. thesis.
	 *
	 * @param[in] h5file A HDF5 file containing the model.
	 * @param[in] landmarkVertexMappingFile A file containing a mapping from landmarks to vertex ids.
	 * @param[in] modelType The type of PCA model to load (SHAPE or COLOR).
	 * @return A shape- or color model from the given file.
	 */
	static PcaModel loadOldBaselH5Model(std::string h5file, std::string landmarkVertexMappingFile, ModelType modelType);

	/**
	 * Load a shape or color model from a .h5 file containing a
	 * statismo-compatible model.
	 *
	 * @param[in] h5file A HDF5 file containing the model.
	 * @param[in] modelType The type of PCA model to load (SHAPE or COLOR).
	 * @return A shape- or color model from the given file.
	 */
	// static PcaModel loadStatismoModel(std::string h5file, ModelType modelType); // TODO!


	/**
	 * Returns the number of principal components in the model.
	 *
	 * @return The number of principal components in the model.
	 */
	unsigned int getNumberOfPrincipalComponents() const;

	/**
	 * Returns the mean of the model.
	 *
	 * @return The mean of the model.
	 */
	cv::Mat getMean() const; // Returning Mesh here makes no sense since the PCA model doesn't know if it's color or shape. Only the MorphableModel can return a Mesh.

	/**
	 * Return the value of the mean at a given landmark.
	 *
	 * @param[in] landmarkIdentifier A landmark identifier (e.g. "center.nose.tip").
	 * @return A Vec3f containing the values at the given landmark.
	 * @throws out_of_range exception if the landmarkIdentifier does not exist in the model. // TODO test the javadoc!
	 */
	cv::Vec3f getMeanAtPoint(std::string landmarkIdentifier) const;

	/**
	 * Return the value of the mean at a given vertex id.
	 *
	 * @param[in] vertexIndex A vertex id.
	 * @return A Vec3f containing the values at the given vertex id.
	 */
	cv::Vec3f getMeanAtPoint(unsigned int vertexIndex) const;

	/**
	 * Draws a random sample from the model, where the coefficients are drawn
	 * from a standard normal (or with the given standard deviation).
	 *
	 * @param[in] sigma The standard deviation. (TODO find out which one, sigma=var, sigmaSq=sdev)
	 * @return A random sample from the model.
	 */
	cv::Mat drawSample(float sigma = 1.0f) const;

	/**
	 * Returns a sample from the model with the given PCA coefficients.
	 *
	 * @param[in] coefficients The PCA coefficients used to generate the sample.
	 * @return A model instance with given coefficients.
	 */
	cv::Mat drawSample(std::vector<float> coefficients) const;

private:
	std::mt19937 engine; ///< A Mersenne twister MT19937 engine
	std::map<std::string, int> landmarkVertexMap; ///< Holds the translation from feature point name (e.g. "center.nose.tip") to the vertex number in the model
	
	cv::Mat mean; ///< A 3m x 1 col-vector (xyzxyz...)', where m is the number of model-vertices
	cv::Mat pcaBasis; // m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
	cv::Mat eigenvalues;

	std::vector<std::array<int, 3>> triangleList; ///< List of triangles that make up the mesh of the model. (Note: Does every PCA model has a triangle-list? Use Mesh here instead?)

};

} /* namespace shapemodels */
#endif /* PCAMODEL_HPP_ */
