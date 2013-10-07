/*
 * MorphableModel.cpp
 *
 *  Created on: 30.09.2013
 *      Author: Patrik Huber
 */

#include "shapemodels/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include <cmath>
#include <iostream> // Todo Replace by logging
#include <fstream>

using cv::Mat;
using cv::Vec4f;

namespace shapemodels {

MorphableModel::MorphableModel()
{
	engine.seed();
}

MorphableModel MorphableModel::load(string h5file, string featurePointsMapping)
{
	MorphableModel mm;
	
	mm.shapeModel.loadFeaturePoints(featurePointsMapping);
	mm.shapeModel.loadModel(h5file, "shape");

	mm.colorModel.loadFeaturePoints(featurePointsMapping);
	mm.colorModel.loadModel(h5file, "color");

	return mm;
}

PcaModel& MorphableModel::getShapeModel()
{
	return shapeModel;
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

MorphableModel MorphableModel::readFromScm(string filename)
{
	MorphableModel mm;
	render::Mesh mesh;

	// Shape:
	unsigned int numVertices = 0;
	unsigned int numTriangles = 0;
	std::vector<unsigned int> triangles;
	unsigned int numShapePcaCoeffs = 0;
	unsigned int numShapeDims = 0;	// what's that?
	std::vector<double> pcaBasisMatrixShp;	// numShapePcaCoeffs * numShapeDims
	unsigned int numMean = 0; // what's that?
	//std::vector<double> meanVertices;
	unsigned int numEigenVals = 0;
	std::vector<double> eigenVals;

	// Texture:
	unsigned int numTexturePcaCoeffs = 0;
	unsigned int numTextureDims = 0;
	std::vector<double> pcaBasisMatrixTex;	// numTexturePcaCoeffs * numTextureDims
	unsigned int numMeanTex = 0; // what's that?
	//std::vector<double> meanVerticesTex; // color mean for each vertex of the mean
	unsigned int numEigenValsTex = 0;
	std::vector<double> eigenValsTex;

	if(sizeof(unsigned int) != 4) {
		std::cout << "Warning: We're reading 4 Bytes from the file but sizeof(unsigned int) != 4. Check the code/behaviour." << std::endl;
	}
	if(sizeof(double) != 8) {
		std::cout << "Warning: We're reading 8 Bytes from the file but sizeof(double) != 8. Check the code/behaviour." << std::endl;
	}

	std::ifstream modelFile;
	modelFile.open(filename, std::ios::binary);
	if (!modelFile.is_open()) {
		std::cout << "Could not open model file: " << filename << std::endl;
		exit(EXIT_FAILURE); // Todo use the logger & stuff
	}

	// make a MM and load both the shape and color models (maybe in another function). For now, we only load the mesh & color info.

	// 1 char = 1 byte. uint32=4bytes. float64=8bytes.

	//READING SHAPE MODEL
	// Read (reference?) num triangles and vertices
	modelFile.read(reinterpret_cast<char*>(&numVertices), 4);
	modelFile.read(reinterpret_cast<char*>(&numTriangles), 4);

	//Read triangles
	mesh.tvi.resize(numTriangles);
	mesh.tci.resize(numTriangles);
	unsigned int v0, v1, v2;
	for (unsigned int i=0; i < numTriangles; ++i) {
		v0 = v1 = v2 = 0;
		modelFile.read(reinterpret_cast<char*>(&v0), 4);	// would be nice to pass a &vector and do it in one
		modelFile.read(reinterpret_cast<char*>(&v1), 4);	// go, but didn't work. Maybe a cv::Mat would work?
		modelFile.read(reinterpret_cast<char*>(&v2), 4);
		mesh.tvi[i][0] = v0;
		mesh.tvi[i][1] = v1;
		mesh.tvi[i][2] = v2;	// do probably the same for tci

		mesh.tci[i][0] = v0;
		mesh.tci[i][1] = v1;
		mesh.tci[i][2] = v2;

	}

	//Read number of rows and columns of the shape projection matrix (pcaBasis)
	modelFile.read(reinterpret_cast<char*>(&numShapePcaCoeffs), 4);
	modelFile.read(reinterpret_cast<char*>(&numShapeDims), 4);

	//Read shape projection matrix
	mm.matPcaBasisShp = cv::Mat(numShapeDims, numShapePcaCoeffs, CV_64FC1);
	// m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
	std::cout << mm.matPcaBasisShp.rows << ", " << mm.matPcaBasisShp.cols << std::endl;
	for (unsigned int col = 0; col < numShapePcaCoeffs; ++col) {
		for (unsigned int row = 0; row < numShapeDims; ++row) {
			double var = 0.0;
			modelFile.read(reinterpret_cast<char*>(&var), 8);
			mm.matPcaBasisShp.at<double>(row, col) = var;
		}
	}

	//Read mean shape vector
	modelFile.read(reinterpret_cast<char*>(&numMean), 4);
	mm.matMeanShp = cv::Mat(numMean, 1, CV_64FC1);
	unsigned int matCounter = 0;
	mesh.vertex.resize(numMean/3);
	double vd0, vd1, vd2;
	for (unsigned int i=0; i < numMean/3; ++i) {
		vd0 = vd1 = vd2 = 0.0;
		modelFile.read(reinterpret_cast<char*>(&vd0), 8);
		modelFile.read(reinterpret_cast<char*>(&vd1), 8);
		modelFile.read(reinterpret_cast<char*>(&vd2), 8);
		//meanVertices.push_back(var);
		mesh.vertex[i].position = cv::Vec4f(vd0, vd1, vd2, 1.0f);

		mm.matMeanShp.at<double>(matCounter, 0) = vd0;
		++matCounter;
		mm.matMeanShp.at<double>(matCounter, 0) = vd1;
		++matCounter;
		mm.matMeanShp.at<double>(matCounter, 0) = vd2;
		++matCounter;
	}

	//Read shape eigenvalues
	modelFile.read(reinterpret_cast<char*>(&numEigenVals), 4);
	mm.matEigenvalsShp = cv::Mat(numEigenVals, 1, CV_64FC1);
	for (unsigned int i=0; i < numEigenVals; ++i) {
		double var = 0.0;
		modelFile.read(reinterpret_cast<char*>(&var), 8);
		eigenVals.push_back(var);
		mm.matEigenvalsShp.at<double>(i, 0) = var;
	}

	//READING TEXTURE MODEL
	//Read number of rows and columns of projection matrix 
	modelFile.read(reinterpret_cast<char*>(&numTexturePcaCoeffs), 4);
	modelFile.read(reinterpret_cast<char*>(&numTextureDims), 4);
	//Read texture projection matrix
	mm.matPcaBasisTex = cv::Mat(numTextureDims, numTexturePcaCoeffs, CV_64FC1);
	std::cout << mm.matPcaBasisTex.rows << ", " << mm.matPcaBasisTex.cols << std::endl;
	for (unsigned int col = 0; col < numTexturePcaCoeffs; ++col) {
		for (unsigned int row = 0; row < numTextureDims; ++row) {
			double var = 0.0;
			modelFile.read(reinterpret_cast<char*>(&var), 8);
			mm.matPcaBasisTex.at<double>(row, col) = var;
		}
	}

	//Read mean texture vector
	modelFile.read(reinterpret_cast<char*>(&numMeanTex), 4);
	mm.matMeanTex = cv::Mat(numMeanTex, 1, CV_64FC1);
	matCounter = 0;
	for (unsigned int i=0; i < numMeanTex/3; ++i) {
		//double var = 0.0;
		vd0 = vd1 = vd2 = 0.0;
		modelFile.read(reinterpret_cast<char*>(&vd0), 8);
		modelFile.read(reinterpret_cast<char*>(&vd1), 8);
		modelFile.read(reinterpret_cast<char*>(&vd2), 8);
		//meanVerticesTex.push_back(var);
		mesh.vertex[i].color = cv::Vec3f(vd0, vd1, vd2);	// order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB

		mm.matMeanTex.at<double>(matCounter, 0) = vd0;
		++matCounter;
		mm.matMeanTex.at<double>(matCounter, 0) = vd1;
		++matCounter;
		mm.matMeanTex.at<double>(matCounter, 0) = vd2;
		++matCounter;
	}

	//Read texture eigenvalues
	modelFile.read(reinterpret_cast<char*>(&numEigenValsTex), 4);
	mm.matEigenvalsTex = cv::Mat(numEigenValsTex, 1, CV_64FC1);
	for (unsigned int i=0; i < numEigenValsTex; ++i) {
		double var = 0.0;
		modelFile.read(reinterpret_cast<char*>(&var), 8);
		eigenValsTex.push_back(var);
		mm.matEigenvalsTex.at<double>(i, 0) = var;
	}

	modelFile.close();

	mesh.hasTexture = false;

	mm.mesh = mesh;
	return mm; // pReference
}

}