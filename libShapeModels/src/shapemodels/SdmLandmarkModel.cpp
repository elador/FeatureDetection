/*
 * SdmLandmarkModel.cpp
 *
 *  Created on: 02.02.2014
 *      Author: Patrik Huber
 */

#include "shapemodels/SdmLandmarkModel.hpp"

#include <fstream>
#include "opencv2/core/core.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/lexical_cast.hpp"

using boost::lexical_cast;

namespace shapemodels {

SdmLandmarkModel::SdmLandmarkModel()
{

}

SdmLandmarkModel::SdmLandmarkModel(cv::Mat meanLandmarks, std::vector<std::string> landmarkIdentifier, std::vector<cv::Mat> regressorData, std::vector<std::shared_ptr<FeatureDescriptorExtractor>> descriptorExtractors)
{
	this->meanLandmarks = meanLandmarks;
	this->landmarkIdentifier = landmarkIdentifier;
	this->regressorData = regressorData;
	this->descriptorExtractors = descriptorExtractors;
}

int SdmLandmarkModel::getNumLandmarks() const
{
	return meanLandmarks.cols / 2;
}

int SdmLandmarkModel::getNumCascadeLevels() const
{
	return regressorData.size();
}

cv::Mat SdmLandmarkModel::getMeanShape() const
{
	return meanLandmarks.clone();
}

cv::Mat SdmLandmarkModel::getRegressorData(int cascadeLevel)
{
	return regressorData[cascadeLevel];
}

std::vector<cv::Point2f> SdmLandmarkModel::getLandmarksAsPoints() const
{
	std::vector<cv::Point2f> landmarks;
	for (int i = 0; i < getNumLandmarks(); ++i) {
		landmarks.push_back({ meanLandmarks.at<float>(i), meanLandmarks.at<float>(i + getNumLandmarks()) });
	}
	return landmarks;
}

void SdmLandmarkModel::save(boost::filesystem::path filename)
{
	std::ofstream file(filename.string());
	file << "#Comment. Date?" << std::endl;
	file << "numLandmarks " << getNumLandmarks() << std::endl;
	for (const auto& id : landmarkIdentifier) {
		file << id << std::endl;
	}
	// write the mean
	for (int i = 0; i < 2 * getNumLandmarks(); ++i) {
		file << getMeanShape().at<float>(i) << std::endl; // not so efficient, clones every time
	}
	file << "numCascadeSteps " << getNumCascadeLevels() << std::endl;
	for (int i = 0; i < getNumCascadeLevels(); ++i) {
		// write the params for this cascade level
		file << "scale " << i << " rows " << getRegressorData(i).rows << " cols " << getRegressorData(i).cols << std::endl;
		file << "descriptorType " << "OpenCVSift" << std::endl;
		file << "descriptorPostprocessing " << "none" << std::endl;
		file << "descriptorParameters " << 0 << std::endl;
		// write the regressor data
		Mat regressor = getRegressorData(i);
		for (int row = 0; row < regressor.rows; ++row) {
			for (int col = 0; col < regressor.cols; ++col) {
				file << regressor.at<float>(row, col) << " ";
			}
			file << std::endl;
		}
	}
	file.close();
	return;
}

shapemodels::SdmLandmarkModel SdmLandmarkModel::load(boost::filesystem::path filename)
{
	SdmLandmarkModel model;
	std::ifstream file(filename.string());
	std::string line;
	vector<string> stringContainer;
	std::getline(file, line); // skip the first line, it's the description
	std::getline(file, line); // numLandmarks 22
	boost::split(stringContainer, line, boost::is_any_of(" "));
	int numLandmarks = lexical_cast<int>(stringContainer[1]);
	// read the mean landmarks
	model.meanLandmarks = Mat(numLandmarks * 2, 1, CV_32FC1);
	// First all the x-coordinates, then all the  y-coordinates.
	for (int i = 0; i < numLandmarks * 2; ++i) {
		std::getline(file, line);
		model.meanLandmarks.at<float>(i, 0) = lexical_cast<float>(line);
	}
	// read the numHogScales
	std::getline(file, line); // numHogScales 5
	boost::split(stringContainer, line, boost::is_any_of(" "));
	int numHogScales = lexical_cast<int>(stringContainer[1]);
	// for every HOG scale, read a header line and then the matrix data
	for (int i = 0; i < numHogScales; ++i) {
		// read the header line
		std::getline(file, line); // scale 1 rows 3169 cols 44
		boost::split(stringContainer, line, boost::is_any_of(" "));
		int numRows = lexical_cast<int>(stringContainer[3]); // = numHogDimensions
		int numCols = lexical_cast<int>(stringContainer[5]); // = numLandmarks * 2
		HogParameter params;
		params.cellSize = lexical_cast<int>(stringContainer[7]); // = cellSize
		params.numBins = lexical_cast<int>(stringContainer[9]); // = numBins
		model.hogParameters.push_back(params);
		Mat regressorData(numRows, numCols, CV_32FC1);
		// read numRows lines
		for (int j = 0; j < numRows; ++j) {
			std::getline(file, line); // float1 float2 float3 ... float44
			boost::split(stringContainer, line, boost::is_any_of(" "));
			for (int col = 0; col < numCols; ++col) { // stringContainer contains one more entry than numCols, but we just skip it, it's a whitespace
				regressorData.at<float>(j, col) = lexical_cast<float>(stringContainer[col]);
			}

		}
		model.regressorData.push_back(regressorData);
	}

	return model;
}

}
