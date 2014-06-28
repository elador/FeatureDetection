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

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"

#include <string>
#include <vector>
#include <array>
#include <map>
#include <random>

namespace morphablemodel {

/**
 * This class represents a PCA-model that consists of:
 *   - a mean vector (y x z)
 *   - a PCA basis matrix (unnormalized and normalized)
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
	 * Note on multi-resolution models: The landmarks to vertex-id mapping is
	 * always the same. The lowest resolution model has all the landmarks defined
	 * and for the higher resolutions, the mesh is divided from that on.
	 * Note: For new landmarks we add, this might not be the case if we add them
	 * in the highest resolution model, so take care!
	 *
	 * - The pcaBasis matrix stored in the file and loaded is the orthogonal PCA basis, i.e. it is not normalized by the eigenvalues.
	 *
	 * @param[in] modelFile A binary .scm-file containing the model.
	 * @param[in] landmarkVertexMappingFile A file containing a mapping from landmarks to vertex ids.
	 * @param[in] modelType The type of PCA model to load (SHAPE or COLOR).
	 * @return A shape- or color model from the given file.
	 */
	static PcaModel loadScmModel(boost::filesystem::path modelFilename, boost::filesystem::path landmarkVertexMappingFile, ModelType modelType);

	/**
	 * Load a shape or color model from a .h5 file containing a
	 * statismo-compatible model.
	 *
	 * Notes: 
	 * - With multi-level models, the reference always has the same (smaller)
	 *   number of vertices than the model
	 * - The landmarks are defined on the reference in l7 and are an exact match. For the lower resolution
	 *   models, the closest approximate vertex in the lower resolution reference is found and stored in the
	 *   model file (at training-time), so every level always contains landmark coordinates that can be exactly
	 *   matched to the reference of the respective level.
	 * - The pcaBasis matrix stored in the file and loaded is already normalized by the eigenvalues.
	 *
	 * @param[in] h5file A HDF5 file containing the model.
	 * @param[in] modelType The type of PCA model to load (SHAPE or COLOR).
	 * @return A shape- or color model from the given file.
	 */
	static PcaModel loadStatismoModel(boost::filesystem::path h5file, ModelType modelType);

	/**
	 * Returns the number of principal components in the model.
	 *
	 * @return The number of principal components in the model.
	 */
	unsigned int getNumberOfPrincipalComponents() const;

	/**
	 * Returns TODO.
	 *
	 * @return TODO.
	 */
	unsigned int getDataDimension() const;

	/**
	 * Returns a list of triangles on how to assemble the vertices into a mesh.
	 *
	 * @return The list of triangles to build a mesh.
	 */
	std::vector<std::array<int, 3>> getTriangleList() const;

	/**
	 * Returns the mean of the model.
	 *
	 * @return The mean of the model.
	 */
	cv::Mat getMean() const; // Returning Mesh here makes no sense since the PCA model doesn't know if it's color or shape. Only the MorphableModel can return a Mesh.

	/**
	 * Return the value of the mean at a given landmark.
	 *
	 * @param[in] landmarkIdentifier A landmark identifier (e.g. "center.nose.tip"). At the moment, this is the vertex id.
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
	 * @param[in] sigma The standard deviation.
	 * @return A random sample from the model.
	 */
	cv::Mat drawSample(float sigma = 1.0f);

	/**
	 * Returns a sample from the model with the given PCA coefficients.
	 *
	 * @param[in] coefficients The PCA coefficients used to generate the sample.
	 * @return A model instance with given coefficients.
	 */
	cv::Mat drawSample(std::vector<float> coefficients);

	/**
	* Returns The PCA basis matrix, i.e. the eigenvectors.
	* Each column of the matrix is an eigenvector.
	* The returned basis is normalized, i.e. every eigenvector
	* is normalized by multiplying it with its eigenvalue.
	* Returns a clone of the matrix so that the original cannot
	* be modified.
	*
	* @return Returns the normalized PCA basis matrix.
	*/
	cv::Mat getNormalizedPcaBasis() const;
	
	/**
	* Returns the PCA basis for a particular vertex. The vertex
	* is specified by a landmark identifier that the model can
	* translate into a vertex number.
	* The returned basis is normalized, i.e. every eigenvector
	* is normalized by multiplying it with its eigenvalue.
	* Returns a clone of the matrix so that the original cannot
	* be modified.
	*
	* @param[in] landmarkIdentifier At the moment, we have to pass the vertex id. TODO: Somehow work with a mapping. Use our new mapper class?
	* @return Todo.
	*/
	cv::Mat getNormalizedPcaBasis(std::string landmarkIdentifier) const;

	/**
	* Returns the PCA basis for a particular vertex. The vertex
	* is specified by a landmark identifier that the model can
	* translate into a vertex number.
	* The returned basis is not normalized, i.e. the eigenvectors
	* are not pre-normalized by multiplying them with their eigenvalues.
	* Returns a clone of the matrix so that the original cannot
	* be modified.
	*
	* @param[in] landmarkIdentifier At the moment, we have to pass the vertex id. TODO: Somehow work with a mapping. Use our new mapper class?
	* @return Todo.
	*/
	cv::Mat getUnnormalizedPcaBasis(std::string landmarkIdentifier) const;

	float getEigenvalue(unsigned int index) const;

	/**
	* Returns true if the given landmark identifier
	* exists in the model.
	*
	* @param[in] landmarkIdentifier A landmark identifier (e.g. "center.nose.tip"). At the moment, this is the vertex id.
	* @return A boolean whether the given landmark identifier exists in the model.
	*/
	bool landmarkExists(std::string landmarkIdentifier) const;

private:
	std::mt19937 engine; ///< A Mersenne twister MT19937 engine
	std::map<std::string, int> landmarkVertexMap; ///< Holds the translation from feature point name (e.g. "center.nose.tip") to the vertex number in the model
	
	cv::Mat mean; ///< A 3m x 1 col-vector (xyzxyz...)', where m is the number of model-vertices.
	cv::Mat normalizedPcaBasis; ///< The normalized PCA basis matrix. m x n (rows x cols) = numShapeDims x numShapePcaCoeffs, (=eigenvector matrix V). Each column is an eigenvector.
	cv::Mat unnormalizedPcaBasis; ///< The unnormalized PCA basis matrix. m x n (rows x cols) = numShapeDims x numShapePcaCoeffs, (=eigenvector matrix V). Each column is an eigenvector.
	
	cv::Mat eigenvalues; ///< A col-vector of the eigenvalues (variances in the PCA space).

	std::vector<std::array<int, 3>> triangleList; ///< List of triangles that make up the mesh of the model. (Note: Does every PCA model has a triangle-list? Use Mesh here instead?)
};

/**
 * Takes an unnormalized PCA basis matrix (a matrix consisting
 * of the eigenvectors and normalizes it, i.e. multiplies each
 * eigenvector by the square root of its corresponding
 * eigenvalue.
 *
 * @param[in] unnormalizedBasis An unnormalized PCA basis matrix.
 * @param[in] eigenvalues A row or column vector of eigenvalues.
 * @return The normalized PCA basis matrix.
 */
cv::Mat normalizePcaBasis(cv::Mat unnormalizedBasis, cv::Mat eigenvalues);

/**
 * Takes a normalized PCA basis matrix (a matrix consisting
 * of the eigenvectors and denormalizes it, i.e. multiplies each
 * eigenvector by 1 over the square root of its corresponding
 * eigenvalue.
 * Note: UNTESTED
 *
 * @param[in] normalizedBasis A normalized PCA basis matrix.
 * @param[in] eigenvalues A row or column vector of eigenvalues.
 * @return The unnormalized PCA basis matrix.
 */
cv::Mat unnormalizePcaBasis(cv::Mat normalizedBasis, cv::Mat eigenvalues);

} /* namespace morphablemodel */
#endif /* PCAMODEL_HPP_ */
