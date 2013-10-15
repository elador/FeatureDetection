/*
 * MorphableModel.cpp
 *
 *  Created on: 30.09.2013
 *      Author: Patrik Huber
 */

#include "shapemodels/MorphableModel.hpp"
/*
#include "opencv2/core/core.hpp"

#include <cmath>
#include <iostream> // Todo Replace by logging
#include <fstream>

using cv::Mat;
using cv::Vec4f;
using std::string;
*/
namespace shapemodels {

MorphableModel::MorphableModel()
{
	
}

shapemodels::MorphableModel MorphableModel::loadOldBaselH5Model(std::string h5file, std::string landmarkVertexMappingFile)
{
	MorphableModel model;
	model.shapeModel = PcaModel::loadOldBaselH5Model(h5file, landmarkVertexMappingFile, PcaModel::ModelType::SHAPE);
	model.colorModel = PcaModel::loadOldBaselH5Model(h5file, landmarkVertexMappingFile, PcaModel::ModelType::COLOR);
	return model;
}

shapemodels::MorphableModel MorphableModel::loadScmModel(std::string h5file, std::string landmarkVertexMappingFile)
{
	MorphableModel model;
	model.shapeModel = PcaModel::loadScmModel(h5file, landmarkVertexMappingFile, PcaModel::ModelType::SHAPE);
	model.colorModel = PcaModel::loadScmModel(h5file, landmarkVertexMappingFile, PcaModel::ModelType::COLOR);
	return model;
}


shapemodels::PcaModel MorphableModel::getShapeModel() const
{
	return shapeModel;
}

shapemodels::PcaModel MorphableModel::getColorModel() const
{
	return colorModel;
}

}