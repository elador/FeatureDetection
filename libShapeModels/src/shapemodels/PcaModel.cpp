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
using cv::Vec3f;
using boost::lexical_cast;
using std::string;
using std::vector;
using std::array;

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
		if (line == "") {
			continue;
		}
		string currFfp; // Have a buffer string
		int currVertex = 0;
		std::stringstream ss(line); // Insert the string into a stream
		ss >> currFfp;
		ss >> currVertex;
		model.landmarkVertexMap.insert(make_pair(currFfp, currVertex));
		currFfp.clear();
	}
	ffpList.close();

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
	model.mean = Mat(1, dims[0], CV_32FC1); // Use a row-vector, because of faster memory access and I'm not sure the memory block is allocated contiguously if we have multiple rows. Maybe change to col-vec later, it's more natural in the calculations.
	dsMean.read(model.mean.ptr<float>(0), H5::PredType::NATIVE_FLOAT);
	model.mean = model.mean.t(); // Okay, transpose it now
	dsMean.close();

	h5Model.close();

	return model;
}



PcaModel PcaModel::loadScmModel(string modelFilename, string landmarkVertexMappingFile, PcaModel::ModelType modelType)
{
	logging::Logger logger = Loggers->getLogger("shapemodels");
	PcaModel model;

	// Load the landmarks mappings
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
		if (line == "" || line.substr(0, 2) == "//") { // empty line or starting with a '//'
			continue;
		}
		string currFfp; // Have a buffer string
		int currVertex = 0;
		std::stringstream ss(line); // Insert the string into a stream
		ss >> currFfp;
		ss >> currVertex;
		model.landmarkVertexMap.insert(make_pair(currFfp, currVertex));
		currFfp.clear();
	}
	ffpList.close();

	// Load the model
	if (sizeof(unsigned int) != 4) {
		logger.warn("Warning: We're reading 4 Bytes from the file but sizeof(unsigned int) != 4. Check the code/behaviour.");
	}
	if (sizeof(double) != 8) {
		logger.warn("Warning: We're reading 8 Bytes from the file but sizeof(double) != 8. Check the code/behaviour.");
	}

	std::ifstream modelFile;
	modelFile.open(modelFilename, std::ios::binary);
	if (!modelFile.is_open()) {
		logger.warn("Could not open model file: " + modelFilename);
		exit(EXIT_FAILURE);
	}

	// Reading the shape model
	// Read (reference?) num triangles and vertices
	unsigned int numVertices = 0;
	unsigned int numTriangles = 0;
	modelFile.read(reinterpret_cast<char*>(&numVertices), 4); // 1 char = 1 byte. uint32=4bytes. float64=8bytes.
	modelFile.read(reinterpret_cast<char*>(&numTriangles), 4);

	// Read triangles
	std::vector<std::array<int, 3>> triangleList;

	triangleList.resize(numTriangles);
	unsigned int v0, v1, v2;
	for (unsigned int i=0; i < numTriangles; ++i) {
		v0 = v1 = v2 = 0;
		modelFile.read(reinterpret_cast<char*>(&v0), 4);	// would be nice to pass a &vector and do it in one
		modelFile.read(reinterpret_cast<char*>(&v1), 4);	// go, but didn't work. Maybe a cv::Mat would work?
		modelFile.read(reinterpret_cast<char*>(&v2), 4);
		triangleList[i][0] = v0;
		triangleList[i][1] = v1;
		triangleList[i][2] = v2;
	}

	// Read number of rows and columns of the shape projection matrix (pcaBasis)
	unsigned int numShapePcaCoeffs = 0;
	unsigned int numShapeDims = 0;	// dimension of the shape vector (3*numVertices)
	modelFile.read(reinterpret_cast<char*>(&numShapePcaCoeffs), 4);
	modelFile.read(reinterpret_cast<char*>(&numShapeDims), 4);

	if (3*numVertices != numShapeDims) {
		logger.warn("Warning: Number of shape dimensions is not equal to three times the number of vertices. Something will probably go wrong during the loading.");
	}

	// Read shape projection matrix
	Mat pcaBasisShape = cv::Mat(numShapeDims, numShapePcaCoeffs, CV_64FC1); // -> to memb.var
	// m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
	logger.debug("Loading PCA basis matrix with " + lexical_cast<string>(pcaBasisShape.rows) + " rows and " + lexical_cast<string>(pcaBasisShape.cols) + "cols.");
	for (unsigned int col = 0; col < numShapePcaCoeffs; ++col) {
		for (unsigned int row = 0; row < numShapeDims; ++row) {
			double var = 0.0;
			modelFile.read(reinterpret_cast<char*>(&var), 8);
			pcaBasisShape.at<double>(row, col) = var;
		}
	}

	// Read mean shape vector
	unsigned int numMean = 0; // dimension of the mean (3*numVertices)
	modelFile.read(reinterpret_cast<char*>(&numMean), 4);
	if (numMean != numShapeDims) {
		logger.warn("Warning: Number of shape dimensions is not equal to the number of dimensions of the mean. Something will probably go wrong during the loading.");
	}
	Mat meanShape = cv::Mat(numMean, 1, CV_32FC1); // -> to memb.var
	unsigned int counter = 0;
	double vd0, vd1, vd2;
	for (unsigned int i=0; i < numMean/3; ++i) {
		vd0 = vd1 = vd2 = 0.0;
		modelFile.read(reinterpret_cast<char*>(&vd0), 8);
		modelFile.read(reinterpret_cast<char*>(&vd1), 8);
		modelFile.read(reinterpret_cast<char*>(&vd2), 8);
		meanShape.at<float>(counter, 0) = vd0;
		++counter;
		meanShape.at<float>(counter, 0) = vd1;
		++counter;
		meanShape.at<float>(counter, 0) = vd2;
		++counter;
	}

	// Read shape eigenvalues
	unsigned int numEigenValsShape = 0;
	modelFile.read(reinterpret_cast<char*>(&numEigenValsShape), 4);
	if (numEigenValsShape != numShapePcaCoeffs) {
		logger.warn("Warning: Number of coefficients in the PCA basis matrix is not equal to the number of eigenvalues. Something will probably go wrong during the loading.");
	}
	Mat eigenvaluesShape = Mat(numEigenValsShape, 1, CV_64FC1); // -> to memb.var
	for (unsigned int i=0; i < numEigenValsShape; ++i) {
		double var = 0.0;
		modelFile.read(reinterpret_cast<char*>(&var), 8);
		eigenvaluesShape.at<double>(i, 0) = var;
	}

	if (modelType == ModelType::SHAPE) {
		model.mean = meanShape;
		model.pcaBasis = pcaBasisShape;
		model.eigenvalues = eigenvaluesShape;
		model.triangleList = triangleList;

		modelFile.close();

		return model;
	}

	// Reading the color model
	// Read number of rows and columns of projection matrix
	unsigned int numTexturePcaCoeffs = 0;
	unsigned int numTextureDims = 0;
	modelFile.read(reinterpret_cast<char*>(&numTexturePcaCoeffs), 4);
	modelFile.read(reinterpret_cast<char*>(&numTextureDims), 4);
	// Read color projection matrix
	Mat pcaBasisColor = cv::Mat(numTextureDims, numTexturePcaCoeffs, CV_64FC1);  // -> to memb.var
	logger.debug("Loading PCA basis matrix with " + lexical_cast<string>(pcaBasisShape.rows) + " rows and " + lexical_cast<string>(pcaBasisShape.cols) + "cols.");
	for (unsigned int col = 0; col < numTexturePcaCoeffs; ++col) {
		for (unsigned int row = 0; row < numTextureDims; ++row) {
			double var = 0.0;
			modelFile.read(reinterpret_cast<char*>(&var), 8);
			pcaBasisColor.at<double>(row, col) = var;
		}
	}

	// Read mean color vector
	unsigned int numMeanColor = 0; // dimension of the mean (3*numVertices)
	modelFile.read(reinterpret_cast<char*>(&numMeanColor), 4);
	Mat meanColor = cv::Mat(numMeanColor, 1, CV_64FC1);  // -> to memb.var
	counter = 0;
	for (unsigned int i=0; i < numMeanColor/3; ++i) {
		vd0 = vd1 = vd2 = 0.0;
		modelFile.read(reinterpret_cast<char*>(&vd0), 8); // order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB
		modelFile.read(reinterpret_cast<char*>(&vd1), 8);
		modelFile.read(reinterpret_cast<char*>(&vd2), 8);
		meanColor.at<double>(counter, 0) = vd0;
		++counter;
		meanColor.at<double>(counter, 0) = vd1;
		++counter;
		meanColor.at<double>(counter, 0) = vd2;
		++counter;
	}

	// Read color eigenvalues
	unsigned int numEigenValsColor = 0;
	modelFile.read(reinterpret_cast<char*>(&numEigenValsColor), 4);
	Mat eigenvaluesColor = cv::Mat(numEigenValsColor, 1, CV_64FC1); // -> to memb.var
	for (unsigned int i=0; i < numEigenValsColor; ++i) {
		double var = 0.0;
		modelFile.read(reinterpret_cast<char*>(&var), 8);
		eigenvaluesColor.at<double>(i, 0) = var;
	}

	if (modelType == ModelType::COLOR) {
		model.mean = meanColor;
		model.pcaBasis = pcaBasisColor;
		model.eigenvalues = eigenvaluesColor;
		model.triangleList = triangleList;

		modelFile.close();

		return model;
	}

	logger.error("Unknown ModelType, should never reach here.");
	//modelFile.close();
	//return model;
}

Mat PcaModel::getMean() const
{
	return mean;
}

Vec3f PcaModel::getMeanAtPoint(string landmarkIdentifier) const
{
	int vertexId = landmarkVertexMap.at(landmarkIdentifier);
	vertexId *= 3;
	return Vec3f(mean.at<float>(vertexId), mean.at<float>(vertexId+1), mean.at<float>(vertexId+2)); // we could use Vec3f(mean(Range(), Range())), maybe then we don't copy the data?
}

Vec3f PcaModel::getMeanAtPoint(unsigned int vertexIndex) const
{
	vertexIndex *= 3;
	return Vec3f(mean.at<float>(vertexIndex), mean.at<float>(vertexIndex+1), mean.at<float>(vertexIndex+2));
}



}