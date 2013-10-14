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
	model.setShapeModel(PcaModel::loadOldBaselH5Model(h5file, landmarkVertexMappingFile, PcaModel::ModelType::SHAPE));
	model.setColorModel(PcaModel::loadOldBaselH5Model(h5file, landmarkVertexMappingFile, PcaModel::ModelType::COLOR));
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

void MorphableModel::setShapeModel(PcaModel shapeModel)
{
	this->shapeModel = shapeModel;
}

void MorphableModel::setColorModel(PcaModel colorModel)
{
	this->colorModel = colorModel;
}

/*
void MorphableModel::drawNewVertexPositions()
{
	//cv::Mat matPcaBasisShp; // m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
	//cv::Mat matMeanShp;
	//cv::Mat matEigenvalsShp;

	//std::uniform_int_distribution<int> distribution(0, 99);
	std::normal_distribution<double> distribution(0.0, 0.8);

	Mat alphas = Mat::zeros(55, 1, CV_64FC1);
	for (int row=0; row < alphas.rows; ++row) {
		alphas.at<double>(row, 0) = distribution(engine);
	}

	//Mat smallBasis = matPcaBasisShp(cv::Rect(0, 0, 55, 100));

	Mat matSqrtEigenvalsShp = matEigenvalsShp.clone();
	for (unsigned int i=0; i<matEigenvalsShp.rows; ++i)	{
		matSqrtEigenvalsShp.at<double>(i) = std::sqrt(matEigenvalsShp.at<double>(i));
	}

	Mat vertices = matMeanShp + matPcaBasisShp * alphas.mul(matSqrtEigenvalsShp);

	unsigned int matIdx = 0;
	for (auto& v : mesh.vertex) {
		v.position = Vec4f(vertices.at<double>(matIdx), vertices.at<double>(matIdx+1), vertices.at<double>(matIdx+2), 1.0f);
		matIdx += 3;
	}

}

void MorphableModel::drawNewVertexPositions(Mat coefficients)
{

	Mat matSqrtEigenvalsShp = matEigenvalsShp.clone();
	for (unsigned int i=0; i<matEigenvalsShp.rows; ++i)	{
		matSqrtEigenvalsShp.at<double>(i) = std::sqrt(matEigenvalsShp.at<double>(i));
	}

	Mat vertices = matMeanShp + matPcaBasisShp * coefficients.mul(matSqrtEigenvalsShp);

	unsigned int matIdx = 0;
	for (auto& v : mesh.vertex) {
		v.position = Vec4f(vertices.at<double>(matIdx), vertices.at<double>(matIdx+1), vertices.at<double>(matIdx+2), 1.0f);
		matIdx += 3;
	}
}
*/
}