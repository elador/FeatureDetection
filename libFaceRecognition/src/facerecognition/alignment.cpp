/*
 * alignment.cpp
 *
 *  Created on: 09.11.2014
 *      Author: Patrik Huber
 */
#include "facerecognition/alignment.hpp"

#include "logging/LoggerFactory.hpp"

#include "opencv2/imgproc/imgproc.hpp"

using logging::LoggerFactory;
using cv::Mat;

namespace facerecognition {

cv::Mat getRotationMatrixFromEyePairs(cv::Vec2f rightEye, cv::Vec2f leftEye)
{
	// Angle calc:
	cv::Vec2f reToLeLandmarksLine(leftEye - rightEye);
	float angle = std::atan2(reToLeLandmarksLine[1], reToLeLandmarksLine[0]);
	float angleDegrees = angle * (180.0 / 3.141592654);
	// IED:
	float ied = cv::norm(reToLeLandmarksLine, cv::NORM_L2);
	// Rotate it:
	cv::Vec2f centerOfRotation = (rightEye + leftEye) / 2; // between the eyes
	return cv::getRotationMatrix2D(centerOfRotation, angleDegrees, 1.0f);
}

cv::Mat cropAligned(cv::Mat image, cv::Vec2f rightEye, cv::Vec2f leftEye, float widthFactor/*=1.1f*/, float heightFactor/*=0.8f*/, float* translationX/*=nullptr*/, float* translationY/*=nullptr*/)
{
	// Crop, place eyes in "middle" horizontal, and at around 1/3 vertical
	cv::Vec2f cropCenter = (rightEye + leftEye) / 2; // between the eyes
	auto ied = cv::norm(leftEye - rightEye, cv::NORM_L2);
	cv::Rect roi(cropCenter[0] - widthFactor * ied, cropCenter[1] - heightFactor * ied, 2 * widthFactor * ied, (heightFactor + 2 * heightFactor) * ied);
	return image(roi);
}

} /* namespace facerecognition */
