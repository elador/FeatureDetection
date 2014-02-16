/*
 * LandmarkBasedSupervisedDescentTraining.cpp
 *
 *  Created on: 04.02.2014
 *      Author: Patrik Huber
 */

#include "shapemodels/LandmarkBasedSupervisedDescentTraining.hpp"
//#include "logging/LoggerFactory.hpp"

#include <fstream>
//#include <cmath>

//using logging::LoggerFactory;
using cv::Mat;
using cv::Rect;
using cv::Scalar;
//using cv::Vec3f;
//using cv::Vec4f;
//using boost::lexical_cast;
//using std::vector;
using std::string;

namespace shapemodels {


void LandmarkBasedSupervisedDescentTraining::saveShapeInstanceToMLtxt(Mat shapeInstance, string filename)
{
	int numLandmarks;
	if (shapeInstance.rows > 1) {
		numLandmarks = shapeInstance.rows / 2;
	}
	else {
		numLandmarks = shapeInstance.cols / 2;
	}
	std::ofstream myfile;
	myfile.open(filename);
	myfile << "x = [";
	for (int i = 0; i < numLandmarks; ++i) {
		myfile << shapeInstance.at<float>(i) << ", ";
	}
	myfile << "];" << std::endl << "y = [";
	for (int i = 0; i < numLandmarks; ++i) {
		myfile << shapeInstance.at<float>(i + numLandmarks) << ", ";
	}
	myfile << "];" << std::endl;
	myfile.close();
}

Mat LandmarkBasedSupervisedDescentTraining::transformLandmarksNormalized(Mat landmarks, Rect box)
{
	Mat transformed(landmarks.rows, landmarks.cols, landmarks.type());
	int numLandmarks = landmarks.cols / 2;
	for (int i = 0; i < numLandmarks; ++i) {
		float normalizedX = ((landmarks.at<float>(i)                -box.x) / static_cast<float>(box.width)) - 0.5f;
		float normalizedY = ((landmarks.at<float>(i + numLandmarks) - box.y) / static_cast<float>(box.height)) - 0.5f;
		transformed.at<float>(i) = normalizedX;
		transformed.at<float>(i + numLandmarks) = normalizedY;
	}
	return transformed;
}

cv::Mat LandmarkBasedSupervisedDescentTraining::meanNormalizationUnitSumSquaredNorms(cv::Mat modelMean)
{
	int numLandmarks = modelMean.cols / 2;
	Mat modelMeanX = modelMean.colRange(0, numLandmarks);
	Mat modelMeanY = modelMean.colRange(numLandmarks, 2 * numLandmarks);
	// calculate the centroid:
	Scalar cx = cv::mean(modelMeanX);
	Scalar cy = cv::mean(modelMeanY);
	// move all points to the centroid
	modelMeanX = modelMeanX - cx[0];
	modelMeanY = modelMeanY - cy[0];
	// scale so that the average norm is 1/numLandmarks (i.e.: the total norm of all vectors added is 1).
	// note: that doesn't make too much sense, because it follows that the more landmarks we use, the smaller the mean-face will be.
	float currentTotalSquaredNorm = 0.0f;
	for (int p = 0; p < numLandmarks; ++p) {
		float x = modelMeanX.at<float>(p);
		float y = modelMeanY.at<float>(p);
		currentTotalSquaredNorm += (x*x + y*y);
	}
	// multiply every vectors coordinate by the sqrt of the currentTotalSquaredNorm
	modelMean /= std::sqrt(currentTotalSquaredNorm);
	return modelMean;
}

cv::Mat LandmarkBasedSupervisedDescentTraining::calculateMean(std::vector<cv::Mat> landmarks, AlignGroundtruth alignGroundtruth, MeanNormalization meanNormalization, std::vector<cv::Rect> faceboxes/*=std::vector<cv::Rect>()*/)
{
	Logger logger = Loggers->getLogger("shapemodels");
	if (landmarks.size() < 1) {
		string msg("No landmarks provided to calculate the mean.");
		logger.error(msg);
		throw std::runtime_error(msg);
	}
	// Calculate the mean-shape of all training images:
	// 1) Prepare the ground-truth landmarks:
	//    Note: We could do some Procrustes and align all shapes to some calculated mean-shape. But actually just the mean calculated like this is a good approximation.
	Mat landmarksMatrix(landmarks.size(), landmarks[0].cols, CV_32FC1);
	switch (alignGroundtruth)
	{
	case AlignGroundtruth::NONE: // from vector<Mat> to one Mat. Just copy, no transformation.
		for (auto currentImage = 0; currentImage < landmarks.size(); ++currentImage) {
			Mat groundtruthLms = landmarks[currentImage];
			groundtruthLms.copyTo(landmarksMatrix.row(currentImage));
		}
		break;
	case AlignGroundtruth::NORMALIZED_FACEBOX:
		if (faceboxes.size() != landmarks.size()) {
			string msg("'AlignGroundtruth' is set to NORMALIZED_FACEBOX but faceboxes.size() != landmarks.size(). Please provide a face-box for each set of landmarks.");
			logger.error(msg);
			throw std::runtime_error(msg);
		}
		for (auto currentImage = 0; currentImage < landmarks.size(); ++currentImage) {
			Mat transformed = transformLandmarksNormalized(landmarks[currentImage], faceboxes[currentImage]);
			transformed.copyTo(landmarksMatrix.row(currentImage));
		}
		break;
	default:
		string msg("'AlignGroundtruth' is set to an unknown value. This should not happen.");
		logger.error(msg);
		throw std::runtime_error(msg);
		break;
	}

	// 2) Take the mean of every row:
	Mat modelMean;
	cv::reduce(landmarksMatrix, modelMean, 0, CV_REDUCE_AVG); // reduce to 1 row

	// 3) Optional: Normalize the calculated mean by something (e.g. make it have a sum of squared norms of 1, as in the original SDM paper)
	switch (meanNormalization)
	{
	case MeanNormalization::NONE:
		break;
	case MeanNormalization::UNIT_SUM_SQUARED_NORMS:
		modelMean = meanNormalizationUnitSumSquaredNorms(modelMean);
		break;
	default:
		string msg("'MeanNormalization' is set to an unknown value. This should not happen.");
		logger.error(msg);
		throw std::runtime_error(msg);
		break;
	}
	saveShapeInstanceToMLtxt(modelMean, "mean.txt");
	return modelMean;
}

}