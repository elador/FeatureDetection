/*
 * PcaModel.cpp
 *
 *  Created on: 30.09.2013
 *      Author: Patrik Huber
 */

#include "shapemodels/PcaModel.hpp"

#include "logging/LoggerFactory.hpp"

#ifdef WIN32	// This is a shitty hack...  find out what the proper way to do this is. Probably include the hdf5.tar.gz in our cmake project. Bzw... without cpp is maybe correct, and my windows-installation is wrong?
	#include "cpp/H5Cpp.h"
#else
	#include "H5Cpp.h"
#endif
//#include "hdf5.h"
#include "boost/lexical_cast.hpp"

#include <fstream>

using logging::LoggerFactory;
using cv::Mat;
using boost::lexical_cast;
using std::string;

namespace shapemodels {

PcaModel::PcaModel()
{
	engine.seed();
}

PcaModel PcaModel::loadOldBaselH5Model(string h5file, string landmarkVertexMappingFile, PcaModel::ModelType modelType)
{
	logging::Logger logger = Loggers->getLogger("shapemodels");
	PcaModel model;

	// Load the landmarks mappings
	std::map<std::string, int> landmarkVertexMap;
	std::ifstream ffpList;
	ffpList.open(landmarkVertexMappingFile.c_str(), std::ios::in);
	if (!ffpList.is_open()) {
		string errorMessage = "Error opening feature points file " + landmarkVertexMappingFile + ".";
		logger.error(errorMessage);
		throw std::runtime_error(errorMessage);
	}
	string line;
	while (ffpList.good()) {
		std::getline(ffpList, line);
		if(line=="") {
			continue;
		}
		string currFfp; // Have a buffer string
		int currVertex = 0;
		std::stringstream ss(line); // Insert the string into a stream
		ss >> currFfp;
		ss >> currVertex;
		landmarkVertexMap.insert(make_pair(currFfp, currVertex));
		currFfp.clear();
	}
	ffpList.close();
	model.setLandmarkVertexMap(landmarkVertexMap);
	

	// Load the shape or color model from the .h5 file
	string h5GroupType;
	if (modelType == ModelType::SHAPE) {
		h5GroupType = "shape";
	} else if (modelType == ModelType::COLOR) {
		h5GroupType = "color";
	}

	H5::H5File h5Model;

	try {
		h5Model = H5::H5File(h5file, H5F_ACC_RDONLY);
	}
	catch (H5::Exception& e) {
		string errorMessage = "Could not open HDF5 file: " + string(e.getCDetailMsg());
		logger.error(errorMessage);
		throw errorMessage;
	}

	// Load either the shape or texture mean
	string h5Group = "/" + h5GroupType + "/ReconstructiveModel/model";
	H5::Group modelReconstructive = h5Model.openGroup(h5Group);

	H5::DataSet dsMean = modelReconstructive.openDataSet("./mean");
	hsize_t dims[1];
	dsMean.getSpace().getSimpleExtentDims(dims, NULL);	// dsMean.getSpace() leaks memory... maybe a hdf5 bug, maybe vlenReclaim(...) could be a fix. No idea.
	//H5::DataSpace dsp = dsMean.getSpace();
	//dsp.close();
	Loggers->getLogger("shapemodels").debug("Dimensions of the model mean: " + lexical_cast<string>(dims[0]));
	Mat modelMean(1, dims[0], CV_32FC1); // Use a row-vector, because of faster memory access and I'm not sure the memory block is allocated contiguously if we have multiple rows. Maybe change to col-vec later, it's more natural in the calculations.
	dsMean.read(modelMean.ptr<float>(0), H5::PredType::NATIVE_FLOAT);
	modelMean = modelMean.t(); // Okay, transpose it now
	dsMean.close();

	model.setMean(modelMean);

	h5Model.close();

	return model;
}



PcaModel PcaModel::loadScmModel(std::string modelFile, std::string landmarkVertexMappingFile, PcaModel::ModelType modelType)
{
	/*
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
	*/
	return PcaModel();
}

void PcaModel::setLandmarkVertexMap(std::map<std::string, int> landmarkVertexMap)
{
	this->landmarkVertexMap = landmarkVertexMap;
}

void PcaModel::setMean( cv::Mat modelMean )
{
	mean = modelMean;
}



}