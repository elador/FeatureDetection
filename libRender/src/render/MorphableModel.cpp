/*
 * MorphableModel.cpp
 *
 *  Created on: 24.06.2013
 *      Author: Patrik Huber
 */

#include "render/MorphableModel.hpp"
#include "opencv2/core/core.hpp"


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
	std::normal_distribution<double> distribution(0.0, 2.0);
	int test = distribution(engine);

	Mat alphas = Mat::zeros(55, 1, CV_64FC1);
	for (int row=0; row < alphas.rows; ++row) {
		alphas.at<double>(row, 0) = distribution(engine);
	}

	Mat vertices = matMeanShp + matPcaBasisShp * alphas;
	Mat smallBasis = matEigenvalsShp(cv::Rect(0, 0, 54, 100));

	unsigned int matIdx = 0;
	for (auto& v : mesh.vertex) {
		v.position = Vec4f(vertices.at<double>(matIdx), vertices.at<double>(matIdx+1), vertices.at<double>(matIdx+2), 1.0f);
		matIdx += 3;
	}

}

} /* namespace render */