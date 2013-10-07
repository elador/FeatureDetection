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
//#include "opencv2/core/core.hpp"

#include <random>

//using cv::Mat;

namespace shapemodels {

/**
 * Desc
 */
class MorphableModel  {
public:

	/**
	 * Constructs a new a.
	 *
	 * @param[in] a b
	 */
	MorphableModel();
	
	/**
	 * Computes the kernel value (dot product in a potentially high dimensional space) of two given vectors.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The kernel value of the two vectors.
	 */
	//virtual double compute(const Mat& lhs, const Mat& rhs) const = 0;

	static MorphableModel load(string h5file, string featurePointsMapping);

	PcaModel& getShapeModel(); // Todo: No ref, but move?

	// The following is from libRender:
	render::Mesh mesh;

	cv::Mat matPcaBasisShp; // m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
	cv::Mat matMeanShp;
	cv::Mat matEigenvalsShp;

	cv::Mat matPcaBasisTex;
	cv::Mat matMeanTex;
	cv::Mat matEigenvalsTex;

	void drawNewVertexPositions();
	void drawNewVertexPositions(cv::Mat coefficients);
	void drawNewVertexColor();
	// End libRender
	static MorphableModel readFromScm(string filename); // from librender::MeshUtils

private:
	PcaModel shapeModel;
	PcaModel colorModel;


	// libRender
	std::mt19937 engine; // Mersenne twister MT19937

};

} /* namespace shapemodels */
#endif /* MORPHABLEMODEL_HPP_ */
