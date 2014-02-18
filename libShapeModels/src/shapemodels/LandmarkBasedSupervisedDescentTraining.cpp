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
	// Todo: Quick doc
	Mat transformed(landmarks.rows, landmarks.cols, landmarks.type());
	int numLandmarks = landmarks.cols / 2;
	for (int i = 0; i < numLandmarks; ++i) { // Todo: This could be done without a for-loop I think, by splitting landmarks in a x- and y-Mat
		float normalizedX = ((landmarks.at<float>(i)                - box.x) / static_cast<float>(box.width)) - 0.5f;
		float normalizedY = ((landmarks.at<float>(i + numLandmarks) - box.y) / static_cast<float>(box.height)) - 0.5f;
		transformed.at<float>(i) = normalizedX;
		transformed.at<float>(i + numLandmarks) = normalizedY;
	}
	return transformed;
}

Mat LandmarkBasedSupervisedDescentTraining::alignMean(Mat mean, Rect faceBox)
{
	// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
	// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
	// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
	Mat alignedMean = mean.clone();
	Mat alignedMeanX = alignedMean.colRange(0, alignedMean.cols / 2);
	Mat alignedMeanY = alignedMean.colRange(alignedMean.cols / 2, alignedMean.cols);
	alignedMeanX = (alignedMeanX + 0.5f) * faceBox.width + faceBox.x;
	alignedMeanY = (alignedMeanY + 0.5f) * faceBox.height + faceBox.y;
	return alignedMean;

	/* with variances: (old code, untested)
	initialShapeEstimate2X0_x = (initialShapeEstimate2X0_x * delta_sx + 0.5f) * detectedFace.width + detectedFace.x + delta_tx;
	initialShapeEstimate2X0_y = (initialShapeEstimate2X0_y * delta_sy + 0.5f) * detectedFace.height + detectedFace.y + delta_ty;
	*/
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

LandmarkBasedSupervisedDescentTraining::ModelVariance LandmarkBasedSupervisedDescentTraining::calculateModelVariance(vector<Mat> trainingImages, vector<cv::Rect> trainingFaceboxes, Mat groundtruthLandmarks, Mat initialShapeEstimateX0)
{
	Mat delta_tx(trainingImages.size(), 1, CV_32FC1);
	Mat delta_ty(trainingImages.size(), 1, CV_32FC1);
	Mat delta_sx(trainingImages.size(), 1, CV_32FC1);
	Mat delta_sy(trainingImages.size(), 1, CV_32FC1);
	int numModelLandmarks = groundtruthLandmarks.cols / 2;
	int numDataPerTrainingImage = initialShapeEstimateX0.rows / groundtruthLandmarks.rows;
	int currentImage = 0;
	for (auto currentImage = 0; currentImage < trainingImages.size(); ++currentImage) {
		Mat img = trainingImages[currentImage];
		cv::Rect detectedFace = trainingFaceboxes[currentImage]; // Caution: Depending on flags selected earlier, we might not have detected faces yet!

		// calculate the centroid and the min-max bounding-box (for the width/height) of the ground-truth and the initial estimate x0:
		Mat groundtruth_x = groundtruthLandmarks.row(currentImage).colRange(0, numModelLandmarks);
		Mat groundtruth_y = groundtruthLandmarks.row(currentImage).colRange(numModelLandmarks, numModelLandmarks * 2);
		
		Mat initialShapeEstimateX0_x = initialShapeEstimateX0.row(currentImage * numDataPerTrainingImage).colRange(0, initialShapeEstimateX0.cols / 2);
		Mat initialShapeEstimateX0_y = initialShapeEstimateX0.row(currentImage * numDataPerTrainingImage).colRange(initialShapeEstimateX0.cols / 2, initialShapeEstimateX0.cols);

		delta_tx.at<float>(currentImage) = calculateTranslationVariance(groundtruth_x, initialShapeEstimateX0_x, detectedFace.width); // This is in relation to the V&J face-box
		delta_ty.at<float>(currentImage) = calculateTranslationVariance(groundtruth_y, initialShapeEstimateX0_y, detectedFace.height);
		
		delta_sx.at<float>(currentImage) = calculateScaleVariance(groundtruth_x, initialShapeEstimateX0_x);
		delta_sy.at<float>(currentImage) = calculateScaleVariance(groundtruth_y, initialShapeEstimateX0_y);

		// Note: The tx we measure is the bias of the face-detector (and should be 0 in an ideal world).
		// We could also use abs(gt-x0)/fb.w instead, what we would then measure is how far the x0 is away
		// from the gt.
		// Actually we should kind of measure both: Correct for the first, and use the latter for the sampling. TODO!!! (MeanNormalization=unitnorm... or FB)
		// However, in the paper they only give variances (for the tracking case), not mu's, assuming the mean is 0 and 1?
		// After long deliberation, I think it's a lost cause - what we measure is just not what is described and what 
		// we can find a reasonable explanation for augmenting training data.
		// Solution: Just try both: Perturb the GT and perturb the x0 (with mu's zero) with the measured variance.
		// NO!!! Perturb the GT by +-mu (mu!=0) makes no sense at all in detection! (maybe in tracking?)
		
		// Wording in paper (?): Calculate the mean and variances of the translational and scaling differences between the initial and true landmark locations:
	}
	// Calculate the mean/variances and store them:
	ModelVariance modelVariance;
	Mat mmu_t_x, mmu_t_y, mmu_s_x, mmu_s_y, msigma_t_x, msigma_t_y, msigma_s_x, msigma_s_y;
	cv::meanStdDev(delta_tx, mmu_t_x, msigma_t_x);
	cv::meanStdDev(delta_ty, mmu_t_y, msigma_t_y);
	cv::meanStdDev(delta_sx, mmu_s_x, msigma_s_x);
	cv::meanStdDev(delta_sy, mmu_s_y, msigma_s_y);
	modelVariance.tx.mu = mmu_t_x.at<double>(0);
	modelVariance.ty.mu = mmu_t_y.at<double>(0);
	modelVariance.sx.mu = mmu_s_x.at<double>(0);
	modelVariance.sy.mu = mmu_s_y.at<double>(0);
	modelVariance.tx.sigma = msigma_t_x.at<double>(0);
	modelVariance.ty.sigma = msigma_t_y.at<double>(0);
	modelVariance.sx.sigma = msigma_s_x.at<double>(0);
	modelVariance.sy.sigma = msigma_s_y.at<double>(0);

	return modelVariance;
}

}