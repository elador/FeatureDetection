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
#include <map>

using cv::Mat;
using std::string;
using std::vector;
using std::map;

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
	 * Computes the kernel value (dot product in a potentially high dimensional space) of two given vectors.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The kernel value of the two vectors.
	 */
	//virtual double compute(const Mat& lhs, const Mat& rhs) const = 0;

	//static PcaModel load(string h5file, string featurePointsMapping);

	void loadModel(string h5file, string h5group);
	void loadFeaturePoints(string filename); // Hmm, we already have something like this in libImageIO, with DidLandmarkMapping etc.

	vector<float>& getMean(); // Todo: No ref, but move?
	map<string, int>& getFeaturePointsMap();

private:

	// All from the old RANSAC code:
	vector<float> modelMeanShp;	// the 3DMM mean shape loaded into memory. Data is XYZXYZXYZ...
	vector<float> modelMeanTex;
	map<string, int> featurePointsMap;	// Holds the translation from feature point name (e.g. reye) to the vertex number in the model



	

};

} /* namespace shapemodels */
#endif /* PCAMODEL_HPP_ */
