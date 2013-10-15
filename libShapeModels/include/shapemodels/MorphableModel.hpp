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

	// drawSample()... get Shp+Col sample, combine, return Mesh?
	
private:
	PcaModel shapeModel; ///< A PCA model over the shape
	PcaModel colorModel; ///< A PCA model over vertex color information

};

} /* namespace shapemodels */
#endif /* MORPHABLEMODEL_HPP_ */
