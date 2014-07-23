/*
 * SdmLandmarkModel.cpp
 *
 *  Created on: 02.02.2014
 *      Author: Patrik Huber
 */

#include "superviseddescent/SdmLandmarkModel.hpp"

#include "imageio/ModelLandmark.hpp"
#include "logging/LoggerFactory.hpp"

#include "opencv2/core/core.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/lexical_cast.hpp"

#include <memory>
#include <fstream>

using logging::Logger;
using logging::LoggerFactory;
using boost::lexical_cast;
using std::shared_ptr;
using std::make_shared;

namespace superviseddescent {

SdmLandmarkModel::SdmLandmarkModel()
{

}

SdmLandmarkModel::SdmLandmarkModel(cv::Mat meanLandmarks, std::vector<std::string> landmarkIdentifier, std::vector<cv::Mat> regressorData, std::vector<std::shared_ptr<DescriptorExtractor>> descriptorExtractors, std::vector<std::string> descriptorTypes)
{
	this->meanLandmarks = meanLandmarks;
	this->landmarkIdentifier = landmarkIdentifier;
	this->regressorData = regressorData;
	this->descriptorExtractors = descriptorExtractors;
	this->descriptorTypes = descriptorTypes;
}

int SdmLandmarkModel::getNumLandmarks() const
{
	return meanLandmarks.cols / 2;
}

int SdmLandmarkModel::getNumCascadeSteps() const
{
	return regressorData.size();
}

cv::Mat SdmLandmarkModel::getMeanShape() const
{
	return meanLandmarks.clone().t();
}

cv::Mat SdmLandmarkModel::getRegressorData(int cascadeLevel)
{
	return regressorData[cascadeLevel];
}

std::shared_ptr<DescriptorExtractor> SdmLandmarkModel::getDescriptorExtractor(int cascadeLevel)
{
	return descriptorExtractors[cascadeLevel];
}

std::string SdmLandmarkModel::getDescriptorType(int cascadeLevel)
{
	return descriptorTypes[cascadeLevel];
}

std::vector<cv::Point2f> SdmLandmarkModel::getMeanAsPoints() const
{
	std::vector<cv::Point2f> landmarks;
	for (int i = 0; i < getNumLandmarks(); ++i) {
		landmarks.push_back({ meanLandmarks.at<float>(i), meanLandmarks.at<float>(i + getNumLandmarks()) });
	}
	return landmarks;
}

cv::Point2f SdmLandmarkModel::getLandmarkAsPoint(std::string landmarkIdentifier, cv::Mat modelInstance/*=cv::Mat()*/) const
{
	const auto it = std::find(begin(this->landmarkIdentifier), end(this->landmarkIdentifier), landmarkIdentifier);
	if (it == end(this->landmarkIdentifier)) {

	} else {
		const auto index = std::distance(begin(this->landmarkIdentifier), it);
		if (modelInstance.empty()) {
			return cv::Point2f(meanLandmarks.at<float>(index), meanLandmarks.at<float>(index + getNumLandmarks()));
		} else {
			return cv::Point2f(modelInstance.at<float>(index), modelInstance.at<float>(index + getNumLandmarks()));
		}
	}
}


void SdmLandmarkModel::save(boost::filesystem::path filename, std::string comment)
{
	std::ofstream file(filename.string());
	file << "# " << comment << std::endl;
	file << "numLandmarks " << getNumLandmarks() << std::endl;
	for (const auto& id : landmarkIdentifier) {
		file << id << std::endl;
	}
	// write the mean
	for (int i = 0; i < 2 * getNumLandmarks(); ++i) {
		file << getMeanShape().at<float>(i) << std::endl; // not so efficient, clones every time
	}
	file << "numCascadeSteps " << getNumCascadeSteps() << std::endl;
	for (int i = 0; i < getNumCascadeSteps(); ++i) {
		// write the params for this cascade level
		file << "cascadeStep " << i << " rows " << getRegressorData(i).rows << " cols " << getRegressorData(i).cols << std::endl;
		file << "descriptorType " << descriptorTypes[i] << std::endl;
		file << "descriptorPostprocessing " << "none" << std::endl;
		file << "descriptorParameters " << descriptorExtractors[i]->getParameterString() << std::endl;
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

SdmLandmarkModel SdmLandmarkModel::load(boost::filesystem::path filename)
{
	Logger logger = Loggers->getLogger("superviseddescent");
	SdmLandmarkModel model;
	std::ifstream file(filename.string());
	if (!file.is_open()) {
		string errorMessage = "Given SDM model file could not be opened: " + filename.string();
		logger.error(errorMessage);
		throw std::runtime_error(errorMessage);
	}
	std::string line;
	vector<string> stringContainer;
	std::getline(file, line); // skip the first line, it's the description
	std::getline(file, line); // numLandmarks 22
	boost::trim_right_if(line, boost::is_any_of("\r")); // Windows line-endings are \r\n, Linux only \n. Thus, when a file has been created on windows and is read on linux, we need to remove the trailing \r.
	boost::split(stringContainer, line, boost::is_any_of(" "));
	int numLandmarks = lexical_cast<int>(stringContainer[1]);
	// read the landmark identifiers:
	for (int i = 0; i < numLandmarks; ++i) {
		std::getline(file, line);
		boost::trim_right_if(line, boost::is_any_of("\r"));
		model.landmarkIdentifier.push_back(line);
	}
	// read the mean landmarks:
	model.meanLandmarks = Mat(1, 2 * numLandmarks, CV_32FC1);
	// First all the x-coordinates, then all the  y-coordinates.
	for (int i = 0; i < numLandmarks * 2; ++i) {
		std::getline(file, line);
		boost::trim_right_if(line, boost::is_any_of("\r"));
		model.meanLandmarks.at<float>(i) = lexical_cast<float>(line);
	}
	// read the numHogScales
	std::getline(file, line); // numCascadeSteps 5
	boost::trim_right_if(line, boost::is_any_of("\r"));
	boost::split(stringContainer, line, boost::is_any_of(" "));
	int numCascadeSteps = lexical_cast<int>(stringContainer[1]);
	// for every cascade step, read 4 header lines and then the matrix data
	for (int i = 0; i < numCascadeSteps; ++i) {
		std::getline(file, line); // cascadeStep 1 rows 3169 cols 44
		boost::trim_right_if(line, boost::is_any_of("\r"));
		boost::split(stringContainer, line, boost::is_any_of(" "));
		int numRows = lexical_cast<int>(stringContainer[3]); // = numFeatureDimensions
		int numCols = lexical_cast<int>(stringContainer[5]); // = numLandmarks * 2

		std::getline(file, line); // descriptorType
		boost::trim_right_if(line, boost::is_any_of("\r"));
		boost::split(stringContainer, line, boost::is_any_of(" "));
		string descriptorType = (stringContainer[1]);
		std::getline(file, line); // descriptorPostprocessing none. Not in use yet.
		std::getline(file, line); // descriptorParameters
		boost::trim_right_if(line, boost::is_any_of("\r"));
		if (descriptorType == "OpenCVSift") { // Todo: make a load method in each descriptor
			shared_ptr<DescriptorExtractor> sift = std::make_shared<SiftDescriptorExtractor>();
			model.descriptorExtractors.push_back(sift);
		}
		else if (descriptorType == "vlhog-dt") {
			boost::split(stringContainer, line, boost::is_any_of(" "));
			stringContainer.erase(stringContainer.begin()); // TODO: CHECK THAT!
			if (stringContainer.size() != 6) {
				throw std::logic_error("descriptorParameters must contain numCells, cellSize and numBins.");
			}
			int numCells = boost::lexical_cast<int>(stringContainer[1]);
			int cellSize = boost::lexical_cast<int>(stringContainer[3]);
			int numBins = boost::lexical_cast<int>(stringContainer[5]);
			shared_ptr<DescriptorExtractor> vlhogDt = std::make_shared<VlHogDescriptorExtractor>(VlHogDescriptorExtractor::VlHogType::DalalTriggs, numCells, cellSize, numBins);
			model.descriptorExtractors.push_back(vlhogDt);
		}
		else if (descriptorType == "vlhog-uoctti") {
			shared_ptr<DescriptorExtractor> vlhogUoctti;
			boost::split(stringContainer, line, boost::is_any_of(" "));
			if (stringContainer.size() == 2) { // use adaptive parameters, depending on the regressor-level and facebox size
				vlhogUoctti = std::make_shared<VlHogDescriptorExtractor>(VlHogDescriptorExtractor::VlHogType::Uoctti);
			}
			else if (stringContainer.size() == 7) { // use the given parameters
				int numCells = boost::lexical_cast<int>(stringContainer[2]);
				int cellSize = boost::lexical_cast<int>(stringContainer[4]);
				int numBins = boost::lexical_cast<int>(stringContainer[6]);
				vlhogUoctti = std::make_shared<VlHogDescriptorExtractor>(VlHogDescriptorExtractor::VlHogType::Uoctti, numCells, cellSize, numBins);
			}
			else {
				throw std::logic_error("descriptorParameters must either be empty (=face-size adaptive parameters) or contain numCells, cellSize and numBins.");
			}
			model.descriptorExtractors.push_back(vlhogUoctti);
		}
		else {
			throw std::logic_error("descriptorType does not match 'OpenCVSift', 'vlhog-dt' or 'vlhog-uoctti'.");
		}

		Mat regressorData(numRows, numCols, CV_32FC1);
		// read numRows lines
		for (int j = 0; j < numRows; ++j) {
			std::getline(file, line); // float1 float2 float3 ... float44
			boost::trim_right_if(line, boost::is_any_of("\r"));
			boost::split(stringContainer, line, boost::is_any_of(" "));
			for (int col = 0; col < numCols; ++col) { // stringContainer contains one more entry than numCols, but we just skip it, it's a whitespace
				regressorData.at<float>(j, col) = lexical_cast<float>(stringContainer[col]);
			}

		}
		model.regressorData.push_back(regressorData);
	}

	return model;
}

imageio::LandmarkCollection SdmLandmarkModel::getAsLandmarks(cv::Mat modelInstance /*= cv::Mat()*/) const
{
	imageio::LandmarkCollection landmarks;
	if (modelInstance.empty()) {
		// a model instance was not provided, return the mean landmarks
		for (size_t i = 0; i < landmarkIdentifier.size(); ++i) {
			landmarks.insert(make_shared<imageio::ModelLandmark>(landmarkIdentifier[i], meanLandmarks.at<float>(i), meanLandmarks.at<float>(i + getNumLandmarks())));
		}
	}
	else {
		// check if the provided model instance is of correct size, then, return it as a LandmarkCollection
		if (modelInstance.rows != getNumLandmarks() * 2) {
			throw std::runtime_error("SdmLandmarkModel: The model instance given has a different number of landmarks than the called model. Please make sure that the instance you give is generated from the model you are calling.");
		}
		for (size_t i = 0; i < landmarkIdentifier.size(); ++i) {
			landmarks.insert(make_shared<imageio::ModelLandmark>(landmarkIdentifier[i], modelInstance.at<float>(i), modelInstance.at<float>(i + getNumLandmarks())));
		}
	}

	return landmarks;
}

}
