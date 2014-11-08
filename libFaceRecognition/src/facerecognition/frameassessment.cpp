/*
 * frameassessment.cpp
 *
 *  Created on: 08.11.2014
 *      Author: Patrik Huber
 */
#include "facerecognition/frameassessment.hpp"

#include "logging/LoggerFactory.hpp"

#include "opencv2/imgproc/imgproc.hpp"

using logging::LoggerFactory;
using cv::Mat;
using std::string;
using std::vector;

namespace facerecognition {

float sharpnessScoreCanny(cv::Mat frame)
{
	// Normalise? (Histo, ZM/UV?) Contrast-normalisation?
	Mat cannyEdges;
	cv::Canny(frame, cannyEdges, 225.0, 175.0); // threshold1, threshold2
	int numEdgePixels = cv::countNonZero(cannyEdges); // throws if 0 nonZero? Check first?

	float sharpness = numEdgePixels * 1000.0f / (cannyEdges.rows * cannyEdges.cols);

	// We'll normalise the sharpness later, per video
	return sharpness;
}

double modifiedLaplacian(const cv::Mat& src)
{
	cv::Mat M = (cv::Mat_<double>(3, 1) << -1, 2, -1);
	cv::Mat G = cv::getGaussianKernel(3, -1, CV_64F);

	cv::Mat Lx;
	cv::sepFilter2D(src, Lx, CV_64F, M, G);

	cv::Mat Ly;
	cv::sepFilter2D(src, Ly, CV_64F, G, M);

	cv::Mat FM = cv::abs(Lx) + cv::abs(Ly);

	double focusMeasure = cv::mean(FM).val[0];
	return focusMeasure;
}

double varianceOfLaplacian(const cv::Mat& src)
{
	cv::Mat lap;
	cv::Laplacian(src, lap, CV_64F);

	cv::Scalar mu, sigma;
	cv::meanStdDev(lap, mu, sigma);

	double focusMeasure = sigma.val[0] * sigma.val[0];
	return focusMeasure;
}

double tenengrad(const cv::Mat& src, int ksize)
{
	cv::Mat Gx, Gy;
	cv::Sobel(src, Gx, CV_64F, 1, 0, ksize);
	cv::Sobel(src, Gy, CV_64F, 0, 1, ksize);

	cv::Mat FM = Gx.mul(Gx) + Gy.mul(Gy);

	double focusMeasure = cv::mean(FM).val[0];
	return focusMeasure;
}

double normalizedGraylevelVariance(const cv::Mat& src)
{
	cv::Scalar mu, sigma;
	cv::meanStdDev(src, mu, sigma);

	double focusMeasure = (sigma.val[0] * sigma.val[0]) / mu.val[0];
	return focusMeasure;
}

std::vector<float> getVideoNormalizedHeadBoxScores(std::vector<float> headBoxSizes)
{
	auto result = std::minmax_element(begin(headBoxSizes), end(headBoxSizes));
	auto min = *result.first;
	auto max = *result.second;

	float m = 1.0f / (max - min);
	float b = -m * min;

	std::transform(begin(headBoxSizes), end(headBoxSizes), begin(headBoxSizes), [m, b](float x) {return m * x + b; });

	return headBoxSizes;
}

std::vector<float> getVideoNormalizedInterEyeDistanceScores(std::vector<float> ieds)
{
	auto result = std::minmax_element(begin(ieds), end(ieds));
	auto min = *result.first;
	auto max = *result.second;

	float m = 1.0f / (max - min);
	float b = -m * min;

	std::transform(begin(ieds), end(ieds), begin(ieds), [m, b](float x) {return m * x + b; });

	return ieds;
}

std::vector<float> getVideoNormalizedYawPoseScores(std::vector<float> yaws)
{
	// We work on the absolute angle values
	std::transform(begin(yaws), end(yaws), begin(yaws), [](float x) {return std::abs(x); });

	// Actually, for the yaw we want an absolute scale and not normalise per video!
	float m = -1.0f / 30.0f; // (30 = 40 - 10) (at 40 = 0, at 10 = 1)
	float b = -40.0f * m; // at 40 = 0
	for (auto& e : yaws) {
		if (e <= 10.0f) {
			e = 1.0f;
		}
		else {
			e = m * e + b;
		}
	}
	return yaws;
}

std::vector<float> getVideoNormalizedCannySharpnessScores(std::vector<float> sharpnesses)
{
	auto result = std::minmax_element(begin(sharpnesses), end(sharpnesses));
	auto min = *result.first;
	auto max = *result.second;

	float m = 1.0f / (max - min);
	float b = -m * min;

	std::transform(begin(sharpnesses), end(sharpnesses), begin(sharpnesses), [m, b](float x) {return m * x + b; });

	return sharpnesses;
}



} /* namespace facerecognition */
