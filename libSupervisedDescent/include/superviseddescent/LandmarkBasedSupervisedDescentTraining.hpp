/*
 * LandmarkBasedSupervisedDescentTraining.hpp
 *
 *  Created on: 04.02.2014
 *      Author: Patrik Huber
 */

#pragma once

#ifndef LANDMARKBASEDSUPERVISEDDESCENTTRAINING_HPP_
#define LANDMARKBASEDSUPERVISEDDESCENTTRAINING_HPP_

#include "superviseddescent/SdmLandmarkModel.hpp"
#include "superviseddescent/DescriptorExtractor.hpp"

#include "opencv2/core/core.hpp"

namespace superviseddescent {

/**
 * A class to train a landmark model based on the supervised descent 
 * method of Xiong and De la Torre, "Supervised Descent Method and its
 * Applications to Face Alignment", CVPR 2013.
 * The class provides reasonable default arguments, so when calling
 * the train function without setting any arguments, it works.
 * However, more detailed stuff can be set by the setters.
 *
 * Todo: Write something about how the landmarks are represented?
 */
class LandmarkBasedSupervisedDescentTraining  {
public:

	/**
	 * Constructs a new LandmarkBasedSupervisedDescentTraining.
	 *
	 * @param[in] a b
	 */
	//LandmarkBasedSupervisedDescentTraining() {};

	struct GaussParameter {
		float mu = 0.0;
		float sigma = 1.0; // Note: sigma = stddev. sigma^2 = variance.
							// Notation is $\mathcal{N}(mu, sigma^2)$ (mean, variance).
							// std::normal_distribution takes (mu, sigma) as arguments.
	};

	// Todo: Note sure what exactly this measures. Think about it.
	struct AlignmentStatistics {
		GaussParameter tx; // translation in x-direction
		GaussParameter ty; // ...
		GaussParameter sx;
		GaussParameter sy;
	};

	// Holds the regularisation parameters for the training.
	struct Regularisation {
		float factor = 0.5f;
		bool regulariseAffineComponent = false;
		bool regulariseWithEigenvalueThreshold = false;
	};

	enum class AlignGroundtruth { // what to do with the GT LMs before mean is taken.
		NONE, // no prealign, stay in img-coords
		NORMALIZED_FACEBOX// translate/scale to facebox, that is a normalized square [-0.5, ...] x ...
	};

	enum class MeanNormalization { // what to do with the mean coords after the mean has been calculated
		NONE,
		UNIT_SUM_SQUARED_NORMS // orig paper
	};

	cv::Mat calculateMean(cv::Mat landmarks, AlignGroundtruth alignGroundtruth, MeanNormalization meanNormalization, std::vector<cv::Rect> faceboxes=std::vector<cv::Rect>());

	// trainingImages: debug only
	// trainingFaceboxes: for normalizing the variances by the face-box
	// groundtruthLandmarks, initialShapeEstimateX0: calc variances
	AlignmentStatistics calculateAlignmentStatistics(std::vector<cv::Rect> trainingFaceboxes, cv::Mat groundtruthLandmarks, cv::Mat initialShapeEstimateX0);

	// Rescale the model-mean by neutralizing it with the current statistics (i.e. dividing by the scale, subtracting transl.) (only necessary if our mean is not normalized to V&J face-box directly in first steps)
	// modifies the input mean!
	cv::Mat rescaleModel(cv::Mat modelMean, const AlignmentStatistics& alignmentStatistics);

	// TODO: Move to MatHelpers::duplicateRows(...)
	Mat duplicateGroundtruthShapes(Mat groundtruthLandmarks, int numSamplesPerImage) {
		Mat groundtruthShapes = Mat::zeros((numSamplesPerImage + 1) * groundtruthLandmarks.rows, groundtruthLandmarks.cols, CV_32FC1); // 10 samples + the original data = 11
		for (int currImg = 0; currImg < groundtruthLandmarks.rows; ++currImg) {
			Mat groundtruthLandmarksRow = groundtruthLandmarks.row(currImg);
			for (int j = 0; j < numSamplesPerImage + 1; ++j) {
				Mat groundtruthShapesRow = groundtruthShapes.row(currImg*(numSamplesPerImage + 1) + j);
				groundtruthLandmarksRow.copyTo(groundtruthShapesRow);
			}
		}
		return groundtruthShapes;
	};

	void setNumSamplesPerImage(int numSamplesPerImage) {
		this->numSamplesPerImage = numSamplesPerImage;
	};

	void setNumCascadeSteps(int numCascadeSteps) {
		this->numCascadeSteps = numCascadeSteps;
	};

	void setRegularisation(Regularisation regularisation) {
		this->regularisation = regularisation;
	};

	void setAlignGroundtruth(AlignGroundtruth alignGroundtruth) {
		this->alignGroundtruth = alignGroundtruth;
	};

	void setMeanNormalization(MeanNormalization meanNormalization) {
		this->meanNormalization = meanNormalization;
	};

public:

	SdmLandmarkModel train(std::vector<cv::Mat> trainingImages, std::vector<cv::Mat> trainingGroundtruthLandmarks, std::vector<cv::Rect> trainingFaceboxes /*maybe optional bzw weglassen hier?*/, std::vector<std::string> modelLandmarks, std::vector<std::string> descriptorTypes, std::vector<std::shared_ptr<DescriptorExtractor>> descriptorExtractors);
	
private:
	int numSamplesPerImage = 10; ///< How many random perturbations to generate per training image
	int numCascadeSteps = 5; ///< How many regressors to train
	Regularisation regularisation; ///< Controls the regularisation of the regressor learning
	AlignGroundtruth alignGroundtruth = AlignGroundtruth::NONE; ///< For mean calc: todo
	MeanNormalization meanNormalization = MeanNormalization::UNIT_SUM_SQUARED_NORMS; ///< F...Mean: todo

	// Transforms one row...
	// Takes the face-box as [-0.5, 0.5] x [-0.5, 0.5] and transforms the landmarks into that rectangle.
	// lms are x1 x2 .. y1 y2 .. row-vec
	cv::Mat transformLandmarksNormalized(cv::Mat landmarks, cv::Rect box);

	// assumes modelMean is row-vec, first half x, second y.
	cv::Mat meanNormalizationUnitSumSquaredNorms(cv::Mat modelMean);
};

/**
 * Below: Free functions / classes belonging to the regression. TODO Move to another file probably. (linearalgebra.hpp? regression.hpp?)
 *
 */

// public?
// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
// - makes a copy of mean, not inplace
// - optional: scaling/trans that gets added to the mean (before scaling up to the facebox)
// Todo/Note: Is this the same as in SdmModel::alignRigid?
cv::Mat alignMean(cv::Mat mean, cv::Rect faceBox, float scalingX = 1.0f, float scalingY = 1.0f, float translationX = 0.0f, float translationY = 0.0f);

// mean translation to go from gt to esti
float calculateMeanTranslation(cv::Mat groundtruth, cv::Mat estimate);

// calc scale ratio of the estimate w.r.t. the GT
// i.e. if the estimate is estimated larger than the GT, it will return > 1.0f
float calculateScaleRatio(cv::Mat groundtruth, cv::Mat estimate);

// deals with both row and col vecs. Assumes first half x, second y.
void saveShapeInstanceToMatlab(cv::Mat shapeInstance, std::string filename);

// todo doc.
void drawLandmarks(cv::Mat image, cv::Mat landmarks, cv::Scalar color = cv::Scalar(0.0, 255.0, 0.0));

// todo doc.
cv::Mat getPerturbedShape(cv::Mat modelMean, LandmarkBasedSupervisedDescentTraining::AlignmentStatistics alignmentStatistics, cv::Rect detectedFace);

/**
 * Todo: Description.
 */
enum class RegularizationType
{
	Manual, ///< use lambda
	Automatic, ///< use norm... optional lambda used as factor
	EigenvalueThreshold ///< see description libEigen
};

/**
 * Todo.
 *
 * @param[in] A The todo.
 * @param[in] b The todo.
 * @param[in] regularizationType Todo.
 * @param[in] lambda For RegularizationType::Automatic: An optional factor to multiply the automatically calculated value with.
 *                   For RegularizationType::Manual: The absolute value of the regularization term.
 * @param[in] regularizeAffineComponent Flag that indicates whether to regularize the affine component as well (the last column(?) of AtA (?)). Default: true
 * @return x.
 */
cv::Mat linearRegression(cv::Mat A, cv::Mat b, RegularizationType regularizationType = RegularizationType::Automatic, float lambda = 0.5f, bool regularizeAffineComponent = true);

/**
* Todo.
*
* @param[in] matrix The todo.
* @return todo.
*/
float calculateEigenvalueThreshold(cv::Mat matrix);

} /* namespace superviseddescent */
#endif /* LANDMARKBASEDSUPERVISEDDESCENTTRAINING_HPP_ */
