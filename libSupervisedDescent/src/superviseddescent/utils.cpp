/*
 * utils.cpp
 *
 *  Created on: 29.07.2014
 *      Author: Patrik Huber
 */
#include "superviseddescent/utils.hpp"

#include "logging/LoggerFactory.hpp"

#include <fstream>

using logging::LoggerFactory;
using cv::Mat;
using std::string;

namespace superviseddescent {

// todo remove stuff and add perturbMean(...). But what about the scaling then, if the sample isn't centered around 0 anymore and we then align it?
Mat alignMean(Mat mean, cv::Rect faceBox, float scalingX/*=1.0f*/, float scalingY/*=1.0f*/, float translationX/*=0.0f*/, float translationY/*=0.0f*/)
{
	// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
	// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
	// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
	Mat alignedMean = mean.clone();
	Mat alignedMeanX = alignedMean.colRange(0, alignedMean.cols / 2);
	Mat alignedMeanY = alignedMean.colRange(alignedMean.cols / 2, alignedMean.cols);
	alignedMeanX = (alignedMeanX*scalingX + 0.5f + translationX) * faceBox.width + faceBox.x;
	alignedMeanY = (alignedMeanY*scalingY + 0.5f + translationY) * faceBox.height + faceBox.y;
	return alignedMean;
}

float calculateMeanTranslation(Mat groundtruth, Mat estimate)
{
	// calculate the centroid of the ground-truth and the estimate
	cv::Scalar gtMean = cv::mean(groundtruth);
	cv::Scalar estMean = cv::mean(estimate);
	// Return the difference between the centroids:
	return (estMean[0] - gtMean[0]);
}

float calculateScaleRatio(Mat groundtruth, Mat estimate)
{
	// calculate the scaling difference between the ground truth and the estimate
	double gtMin, gtMax;
	cv::minMaxIdx(groundtruth, &gtMin, &gtMax);
	double x0Min, x0Max;
	cv::minMaxIdx(estimate, &x0Min, &x0Max);

	return (x0Max - x0Min) / (gtMax - gtMin);
}

void saveShapeInstanceToMatlab(Mat shapeInstance, string filename)
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

void drawLandmarks(cv::Mat image, cv::Mat landmarks, cv::Scalar color /*= cv::Scalar(0.0, 255.0, 0.0)*/)
{
	auto numLandmarks = std::max(landmarks.cols, landmarks.rows) / 2;
	for (int i = 0; i < numLandmarks; ++i) {
		cv::circle(image, cv::Point2f(landmarks.at<float>(i), landmarks.at<float>(i + numLandmarks)), 2, color);
	}
}

} /* namespace superviseddescent */
