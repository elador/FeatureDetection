/*
 * LinearShapeFitting.cpp
 *
 *  Created on: 22.05.2014
 *      Author: Patrik Huber
 */

#include "fitting/LinearShapeFitting.hpp"

#include "logging/LoggerFactory.hpp"

using logging::LoggerFactory;
using morphablemodel::MorphableModel;
using cv::Mat;
using std::vector;

namespace fitting {

vector<float> fitShapeToLandmarksLinear(MorphableModel morphableModel, Mat affineCameraMatrix, vector<imageio::ModelLandmark> landmarks, float lambda/*=20.0f*/, boost::optional<int> numCoefficientsToFit/*=boost::optional<int>()*/, boost::optional<float> detectorStandardDeviation/*=boost::optional<float>()*/, boost::optional<float> modelStandardDeviation/*=boost::optional<float>()*/)
{
	// Not used yet
	//int numCoeffsToFit = numCoefficientsToFit.get_value_or(morphableModel.getShapeModel().getNumberOfPrincipalComponents());

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	Mat V_hat_h = Mat::zeros(4 * landmarks.size(), morphableModel.getShapeModel().getNumberOfPrincipalComponents(), CV_32FC1);
	int rowIndex = 0;
	for (const auto& lm : landmarks) {
		Mat basisRows = morphableModel.getShapeModel().getNormalizedPcaBasis(lm.getName()); // In the paper, the not-normalized basis might be used? I'm not sure, check it. It's even a mess in the paper. PH 26.5.2014: I think the normalized basis is fine/better.
		basisRows.copyTo(V_hat_h.rowRange(rowIndex, rowIndex + 3));
		rowIndex += 4; // replace 3 rows and skip the 4th one, it has all zeros
	}
	// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affineCam) is placed on the diagonal:
	Mat P = Mat::zeros(3 * landmarks.size(), 4 * landmarks.size(), CV_32FC1);
	for (int i = 0; i < landmarks.size(); ++i) {
		Mat submatrixToReplace = P.colRange(4 * i, (4 * i) + 4).rowRange(3 * i, (3 * i) + 3);
		affineCameraMatrix.copyTo(submatrixToReplace);
	}
	// The variances: Add the 2D and 3D standard deviations.
	// If the user doesn't provide them, we choose the following:
	// 2D (detector) variance: Assuming the detector has a standard deviation of 3 pixels, and the face size (IED) is around 80px. That's 3.75% of the IED. Assume that an image is on average 512x512px so 80/512 = 0.16 is the size the IED occupies inside an image.
	//                         Now we're in clip-coords ([-1, 1]) and take 0.16 of the range [-1, 1], 0.16/2 = 0.08, and then the standard deviation of the detector is 3.75% of 0.08, i.e. 0.0375*0.08 = 0.003.
	// 3D (model) variance: 0.0f. It only makes sense to set it to something when we have a different variance for different vertices.
	float sigma_2D_3D = detectorStandardDeviation.get_value_or(0.003f) + modelStandardDeviation.get_value_or(0.0f);
	// Note: Isn't it a bit strange to add those as they have different units/normalizations? Check the paper.
	Mat Sigma = Mat::zeros(3 * landmarks.size(), 3 * landmarks.size(), CV_32FC1);
	for (int i = 0; i < 3 * landmarks.size(); ++i) {
		Sigma.at<float>(i, i) = 1.0f / sigma_2D_3D; // the higher the sigma_2D_3D, the smaller the diagonal entries of Sigma will be
	}
	Mat Omega = Sigma.t() * Sigma; // just squares the diagonal
	// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
	Mat y = Mat::ones(3 * landmarks.size(), 1, CV_32FC1);
	for (int i = 0; i < landmarks.size(); ++i) {
		y.at<float>(3 * i, 0) = landmarks[i].getX();
		y.at<float>((3 * i) + 1, 0) = landmarks[i].getY();
		//y.at<float>((3 * i) + 2, 0) = 1; // already 1, stays (homogeneous coordinate)
	}
	// The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
	Mat v_bar = Mat::ones(4 * landmarks.size(), 1, CV_32FC1);
	for (int i = 0; i < landmarks.size(); ++i) {
		cv::Vec3f modelMean = morphableModel.getShapeModel().getMeanAtPoint(landmarks[i].getName());
		v_bar.at<float>(4 * i, 0) = modelMean[0];
		v_bar.at<float>((4 * i) + 1, 0) = modelMean[1];
		v_bar.at<float>((4 * i) + 2, 0) = modelMean[2];
		//v_bar.at<float>((4 * i) + 3, 0) = 1; // already 1, stays (homogeneous coordinate)
	}

	// Bring into standard regularised quadratic form with diagonal distance matrix Omega
	Mat A = P * V_hat_h; // camera matrix times the basis
	Mat b = P * v_bar - y; // camera matrix times the mean, minus the landmarks.
	//Mat c_s; // The x, we solve for this! (the variance-normalized shape parameter vector, $c_s = [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$
	int numShapePc = morphableModel.getShapeModel().getNumberOfPrincipalComponents();
	Mat AtOmegaA = A.t() * Omega * A;
	Mat AtOmegaAReg = AtOmegaA + lambda * Mat::eye(numShapePc, numShapePc, CV_32FC1);
	// Invert using OpenCV:
	Mat AtOmegaARegInv = AtOmegaAReg.inv(cv::DECOMP_SVD); // DECOMP_SVD calculates the pseudo-inverse if the matrix is not invertible.
	// Invert (and perform some sanity checks) using Eigen:
	/*
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> AtOmegaAReg_Eigen(AtOmegaAReg.ptr<float>(), AtOmegaAReg.rows, AtOmegaAReg.cols);
	Eigen::FullPivLU<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> luOfAtOmegaAReg(AtOmegaAReg_Eigen); // Calculate the full-pivoting LU decomposition of the regularized AtA. Note: We could also try FullPivHouseholderQR if our system is non-minimal (i.e. there are more constraints than unknowns).
	float rankOfAtOmegaAReg = luOfAtOmegaAReg.rank();
	bool isAtOmegaARegInvertible = luOfAtOmegaAReg.isInvertible();
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> AtARegInv_EigenFullLU = luOfAtOmegaAReg.inverse();
	Mat AtOmegaARegInvFullLU(AtARegInv_EigenFullLU.rows(), AtARegInv_EigenFullLU.cols(), CV_32FC1, AtARegInv_EigenFullLU.data()); // create an OpenCV Mat header for the Eigen data
	*/
	Mat AtOmegatb = A.t() * Omega.t() * b;
	Mat c_s = -AtOmegaARegInv * AtOmegatb; // Note/Todo: We get coefficients ~ N(0, sigma) I think. They are not multiplied with the eigenvalues.
	
	return vector<float>(c_s);
}

} /* namespace fitting */
