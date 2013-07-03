/*
 * MorphableModel.hpp
 *
 *  Created on: 24.06.2013
 *      Author: Patrik Huber
 */

#ifndef MORPHABLEMODEL_HPP_
#define MORPHABLEMODEL_HPP_

#include "render/Mesh.hpp"

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

	std::vector<Vertex> drawNewVertexPositions(Mesh mesh);
	std::vector<Vertex> drawNewVertexColor(Mesh mesh);

private:

};

} /* namespace render */

#endif /* MORPHABLEMODEL_HPP_ */
