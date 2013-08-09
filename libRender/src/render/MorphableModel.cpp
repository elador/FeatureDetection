/*
 * MorphableModel.cpp
 *
 *  Created on: 24.06.2013
 *      Author: Patrik Huber
 */

#include "render/MorphableModel.hpp"
#include "opencv2/core/core.hpp"

#include <cmath>

using cv::Mat;
using cv::Vec4f;

namespace render {



MorphableModel::MorphableModel()
{
	engine.seed();
}

MorphableModel::~MorphableModel()
{

}

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

} /* namespace render */