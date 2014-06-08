/*
 * MorphableModel.cpp
 *
 *  Created on: 30.09.2013
 *      Author: Patrik Huber
 */

#include "morphablemodel/MorphableModel.hpp"
#include "logging/LoggerFactory.hpp"
#include "opencv2/core/core.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/filesystem/path.hpp"
#include <exception>
#include <sstream>
#include <fstream>

using logging::LoggerFactory;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using boost::lexical_cast;
using boost::filesystem::path;
using std::vector;
using std::string;

namespace morphablemodel {

MorphableModel::MorphableModel()
{
	
}

morphablemodel::MorphableModel MorphableModel::load(const boost::property_tree::ptree configTree)
{
	MorphableModel morphableModel;
	path filename = configTree.get<path>("filename");
	if (filename.extension().string() == ".scm") {
		path vertexMappingFile = configTree.get<path>("vertexMapping");
		path isomapFile = configTree.get<path>("isomap", "");
		morphableModel = MorphableModel::loadScmModel(filename.string(), vertexMappingFile, isomapFile);
	}
	else if (filename.extension().string() == ".h5") {
		morphableModel = MorphableModel::loadStatismoModel(filename.string());
	}
	else
	{
		throw std::runtime_error("MorphableModel: Unknown file extension. Neither .scm nor .h5.");
	}
	return morphableModel;
}

MorphableModel MorphableModel::loadScmModel(path h5file, path landmarkVertexMappingFile, path isomapFile)
{
	MorphableModel model;
	model.shapeModel = PcaModel::loadScmModel(h5file, landmarkVertexMappingFile, PcaModel::ModelType::SHAPE);
	model.colorModel = PcaModel::loadScmModel(h5file, landmarkVertexMappingFile, PcaModel::ModelType::COLOR);

	if (!isomapFile.empty()) {
		vector<Vec2f> texCoords = MorphableModel::loadIsomap(isomapFile);
		if (model.shapeModel.getDataDimension() / 3.0f != texCoords.size()) {
			// TODO Warn/Error, texCoords will not be used
		}
		else
		{
			model.textureCoordinates = texCoords;
			model.hasTextureCoordinates = true;
		}	
	}
	return model;
}

MorphableModel MorphableModel::loadStatismoModel(path h5file)
{
	MorphableModel model;
	model.shapeModel = PcaModel::loadStatismoModel(h5file, PcaModel::ModelType::SHAPE);
	model.colorModel = PcaModel::loadStatismoModel(h5file, PcaModel::ModelType::COLOR);
	return model;
}


PcaModel MorphableModel::getShapeModel() const
{
	return shapeModel;
}

PcaModel MorphableModel::getColorModel() const
{
	return colorModel;
}

render::Mesh MorphableModel::getMean() const
{
	render::Mesh mean;
	
	mean.tvi = shapeModel.getTriangleList();
	mean.tci = colorModel.getTriangleList();
	
	Mat shapeMean = shapeModel.getMean();
	Mat colorMean = colorModel.getMean();

	unsigned int numVertices = shapeModel.getDataDimension() / 3;
	unsigned int numVerticesColor = colorModel.getDataDimension() / 3;
	if (numVertices != numVerticesColor) {
		// TODO throw more meaningful error, maybe log
		throw std::runtime_error("numVertices should be equal to numVerticesColor.");
	}

	// Construct the mesh vertices
	mean.vertex.resize(numVertices);
	for (unsigned int i = 0; i < numVertices; ++i) {
		mean.vertex[i].position = Vec4f(shapeMean.at<float>(i*3 + 0), shapeMean.at<float>(i*3 + 1), shapeMean.at<float>(i*3 + 2), 1.0f);
		mean.vertex[i].color = Vec3f(colorMean.at<float>(i*3 + 0), colorMean.at<float>(i*3 + 1), colorMean.at<float>(i*3 + 2));        // order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB
	}

	if (hasTextureCoordinates) {
		for (unsigned int i = 0; i < numVertices; ++i) {
			mean.vertex[i].texcrd = textureCoordinates[i];
		}
		
	}
	
	mean.hasTexture = false; // hasTexture is not the same as hasTextureCoordinates...

	return mean;
}

render::Mesh MorphableModel::drawSample(float sigma /*= 1.0f*/)
{
	render::Mesh mean;

	mean.tvi = shapeModel.getTriangleList();
	mean.tci = colorModel.getTriangleList();

	Mat shapeMean = shapeModel.drawSample(sigma);
	Mat colorMean = colorModel.drawSample(sigma);

	unsigned int numVertices = shapeModel.getDataDimension() / 3;
	unsigned int numVerticesColor = colorModel.getDataDimension() / 3;
	if (numVertices != numVerticesColor) {
		string msg("MorphableModel: The number of vertices of the shape and color models are not the same: " + lexical_cast<string>(numVertices) + " != " + lexical_cast<string>(numVerticesColor));
		Loggers->getLogger("shapemodels").debug(msg);
		throw std::runtime_error(msg);
	}

	mean.vertex.resize(numVertices);

	for (unsigned int i = 0; i < numVertices; ++i) {
		mean.vertex[i].position = Vec4f(shapeMean.at<float>(i*3 + 0), shapeMean.at<float>(i*3 + 1), shapeMean.at<float>(i*3 + 2), 1.0f);
		mean.vertex[i].color = Vec3f(colorMean.at<float>(i*3 + 0), colorMean.at<float>(i*3 + 1), colorMean.at<float>(i*3 + 2));        // order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB
	}

	mean.hasTexture = false;

	return mean;
}

render::Mesh MorphableModel::drawSample(vector<float> shapeCoefficients, vector<float> colorCoefficients)
{
	render::Mesh sample;

	sample.tvi = shapeModel.getTriangleList();
	sample.tci = colorModel.getTriangleList();

	Mat shapeSample;
	Mat colorSample;

	if (shapeCoefficients.empty()) {
		shapeSample = shapeModel.getMean();
	} else {
		shapeSample = shapeModel.drawSample(shapeCoefficients);
	}
	if (colorCoefficients.empty()) {
		colorSample = colorModel.getMean();
	} else {
		colorSample = colorModel.drawSample(colorCoefficients);
	}

	unsigned int numVertices = shapeModel.getDataDimension() / 3;
	unsigned int numVerticesColor = colorModel.getDataDimension() / 3;
	if (numVertices != numVerticesColor) {
		string msg("MorphableModel: The number of vertices of the shape and color models are not the same: " + lexical_cast<string>(numVertices) + " != " + lexical_cast<string>(numVerticesColor));
		Loggers->getLogger("morphablemodel").debug(msg);
		throw std::runtime_error(msg);
	}

	sample.vertex.resize(numVertices);

	for (unsigned int i = 0; i < numVertices; ++i) {
		sample.vertex[i].position = Vec4f(shapeSample.at<float>(i*3 + 0), shapeSample.at<float>(i*3 + 1), shapeSample.at<float>(i*3 + 2), 1.0f);
		sample.vertex[i].color = Vec3f(colorSample.at<float>(i*3 + 0), colorSample.at<float>(i*3 + 1), colorSample.at<float>(i*3 + 2));        // order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB
		if (hasTextureCoordinates) {
			sample.vertex[i].texcrd = textureCoordinates[i];
		}
	}

	if (hasTextureCoordinates) {
		sample.hasTexture = true; // Note: Actually it doesn't have a texture, just texture coordinates!
	}

	return sample;
}

vector<Vec2f> MorphableModel::loadIsomap(path isomapFile)
{
	vector<float> xCoords, yCoords;
	string line;
	std::ifstream myfile(isomapFile.string());
	if (!myfile.is_open()) {
		//TODO log "Isomap file could not be opened. Did you specify a correct filename?" isomapFile
	} else {
		while (getline(myfile, line))
		{
			std::istringstream iss(line);
			string x, y;
			iss >> x >> y;
			xCoords.push_back(lexical_cast<float>(x));
			yCoords.push_back(lexical_cast<float>(y));
		}
		myfile.close();
	}
	// Process the coordinates: Find the min/max and rescale to [0, 1] x [0, 1]
	auto minMaxX = std::minmax_element(begin(xCoords), end(xCoords)); // minMaxX is a pair, first=min, second=max
	auto minMaxY = std::minmax_element(begin(yCoords), end(yCoords));

	vector<Vec2f> texCoords;
	float divisorX = *minMaxX.second - *minMaxX.first;
	float divisorY = *minMaxY.second - *minMaxY.first;
	for (int i = 0; i < xCoords.size(); ++i) {
		texCoords.push_back(Vec2f((xCoords[i]-*minMaxX.first)/divisorX, 1.0f - (yCoords[i]-*minMaxY.first)/divisorY)); // We rescale to [0, 1] and at the same time flip the y-coords (because in the isomap, the coordinates are stored upside-down).
	}

	return texCoords;
}

/*void MorphableModel::setHasTextureCoordinates(bool hasTextureCoordinates)
{
	this->hasTextureCoordinates = hasTextureCoordinates;
}*/

/*
unsigned int matIdx = 0;
for (auto& v : mesh.vertex) {
v.position = Vec4f(modelSample.at<double>(matIdx), modelSample.at<double>(matIdx+1), modelSample.at<double>(matIdx+2), 1.0f);
matIdx += 3;
}
*/

} /* namespace morphablemodel */