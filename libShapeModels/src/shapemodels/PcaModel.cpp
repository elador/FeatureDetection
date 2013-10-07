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
using boost::lexical_cast;

namespace shapemodels {


void PcaModel::loadModel(string h5file, string h5group)
{

	H5::H5File h5Model;

	try {
		h5Model = H5::H5File(h5file, H5F_ACC_RDONLY);
	}
	catch (H5::Exception& e) {
		std::string msg( std::string( "Could not open HDF5 file \n" ) + e.getCDetailMsg() ); // TODO Logger
		throw msg;
	}

	// Load the Shape
	H5::Group modelReconstructive = h5Model.openGroup("/shape/ReconstructiveModel/model"); // TODO h5group
	H5::DataSet dsMean = modelReconstructive.openDataSet("./mean");
	hsize_t dims[1];
	dsMean.getSpace().getSimpleExtentDims(dims, NULL);	// dsMean.getSpace() leaks memory... maybe a hdf5 bug, maybe vlenReclaim(...) could be a fix. No idea.
	//H5::DataSpace dsp = dsMean.getSpace();
	//dsp.close();

	Loggers->getLogger("shapemodels").debug("Dimensions: " + lexical_cast<string>(dims[0]));

	float* testData = new float[dims[0]]; // TODO: I guess this whole part could be done A LOT better!
	dsMean.read(testData, H5::PredType::NATIVE_FLOAT);
	this->modelMeanShp.reserve(dims[0]);

	for (unsigned int i=0; i < dims[0]; ++i)	{
		modelMeanShp.push_back(testData[i]);
	}
	delete[] testData;
	testData = NULL;
	dsMean.close();

	// // Load the Texture
	H5::Group modelReconstructiveTex = h5Model.openGroup("/color/ReconstructiveModel/model");
	H5::DataSet dsMeanTex = modelReconstructiveTex.openDataSet("./mean");
	hsize_t dimsTex[1];
	dsMeanTex.getSpace().getSimpleExtentDims(dimsTex, NULL);
	Loggers->getLogger("shapemodels").debug("Dimensions: " + lexical_cast<string>(dimsTex[0]));
	float* testDataTex = new float[dimsTex[0]]; // TODO: I guess this whole part could be done A LOT better!
	dsMeanTex.read(testDataTex, H5::PredType::NATIVE_FLOAT);
	this->modelMeanTex.reserve(dimsTex[0]);

	for (unsigned int i=0; i < dimsTex[0]; ++i)	{
		modelMeanTex.push_back(testDataTex[i]);
	}
	delete[] testDataTex;
	testDataTex = NULL;
	dsMeanTex.close();

	h5Model.close();
	

}


void PcaModel::loadFeaturePoints(string filename)
{
	
	std::ifstream ffpList;
	ffpList.open(filename.c_str(), std::ios::in);
	if (!ffpList.is_open()) {
		Loggers->getLogger("shapemodels").error("Error opening feature points file " + filename + ".");
		exit(EXIT_FAILURE); // TODO replace by throwing
	}
	std::string line;
	while (ffpList.good()) {
		std::getline(ffpList, line);
		if(line=="") {
			continue;
		}
		std::string currFfp; // Have a buffer string
		int currVertex = 0;
		std::stringstream ss(line); // Insert the string into a stream
		ss >> currFfp;
		ss >> currVertex;
		//filenames.push_back(buf);
		featurePointsMap.insert(std::map<std::string, int>::value_type(currFfp, currVertex));
		currFfp.clear();
	}
	ffpList.close();
}

vector<float>& PcaModel::getMean()
{
	return modelMeanShp;
}

map<string, int>& PcaModel::getFeaturePointsMap()
{
	return featurePointsMap;
}


}