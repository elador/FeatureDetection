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

//#include "render/Mesh.hpp"
//#include "opencv2/core/core.hpp"


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

	static MorphableModel loadOldBaselH5Model(std::string h5file, std::string landmarkVertexMappingFile);
	

	// The following is from libRender:
	//render::Mesh mesh;
	/*
	void drawNewVertexPositions();
	void drawNewVertexPositions(cv::Mat coefficients);
	void drawNewVertexColor();*/
	// End libRender
	
	void setShapeModel(PcaModel shapeModel);

	void setColorModel(PcaModel colorModel);

private:
	PcaModel shapeModel;
	PcaModel colorModel;

};

} /* namespace shapemodels */
#endif /* MORPHABLEMODEL_HPP_ */
