/*
 * MorphableModel.hpp
 *
 *  Created on: 24.06.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef MORPHABLEMODEL_HPP_
#define MORPHABLEMODEL_HPP_

#include "render/Mesh.hpp"

#include <random>

namespace render {

/**
 * Desc
 */
class MorphableModel {
public:

	/**
	 * Constructs a new a.
	 *
	 * @param[in] a b
	 */
	MorphableModel();

	virtual ~MorphableModel();

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

private:
	std::mt19937 engine; // Mersenne twister MT19937

};

} /* namespace render */

#endif /* MORPHABLEMODEL_HPP_ */
