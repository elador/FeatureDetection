/*
 * AffineCameraEstimation.cpp
 *
 *  Created on: 31.12.2013
 *      Author: Patrik Huber
 */

#include "fitting/AffineCameraEstimation.hpp"

#include "render/utils.hpp"
#include "logging/LoggerFactory.hpp"

#include "opencv2/core/core_c.h" // for CV_REDUCE_AVG

#include <exception>

using logging::LoggerFactory;
using morphablemodel::MorphableModel;
using cv::Mat;
using cv::Vec3f;
using cv::Vec4f;
using std::vector;

namespace fitting {

Mat estimateAffineCamera(vector<imageio::ModelLandmark> imagePoints, MorphableModel morphableModel, vector<int> vertexIds/*=std::vector<int>()*/)
{
	// Note/TODO: If this function is called with invalid imagePoints, i.e. landmark id's that are not in the 3DMM, nothing throws. Something, somewhere, should happen.
	// Todo: Currently, the optional vertexIds are not used

	Mat matImagePoints; // will be numCorrespondences x 2, CV_32FC1
	Mat matModelPoints; // will be numCorrespondences x 3, CV_32FC1
	// Use the point only if it is available in the model:
	for (const auto& landmark : imagePoints) {
		Vec3f tmp;
		try {
			tmp = morphableModel.getShapeModel().getMeanAtPoint(landmark.getName());
		}
		catch (std::out_of_range& e) {
			continue;
		}
		Mat imgPoint(1, 2, CV_32FC1);
		imgPoint.at<float>(0, 0) = landmark.getX();
		imgPoint.at<float>(0, 1) = landmark.getY();
		matImagePoints.push_back(imgPoint);

		Mat mdlPoint(1, 3, CV_32FC1);
		mdlPoint.at<float>(0, 0) = tmp[0];
		mdlPoint.at<float>(0, 1) = tmp[1];
		mdlPoint.at<float>(0, 2) = tmp[2];
		matModelPoints.push_back(mdlPoint);
	}
	const auto numCorrespondences = matImagePoints.rows;
	if (numCorrespondences < 4) {
		Loggers->getLogger("fitting").error("AffineCameraEstimation: Number of points given needs to be equal to or larger than 4.");
		throw std::runtime_error("AffineCameraEstimation: Number of points given needs to be equal to or larger than 4.");
	}

	//Mat tmpOrigImgPoints = matImagePoints.clone(); // Temp for testing: Save the original coordinates.
	// translate the centroid of the image points to the origin:
	Mat imagePointsMean; // use non-homogeneous coords for the next few steps? (less submatrices etc overhead)
	cv::reduce(matImagePoints, imagePointsMean, 0, CV_REDUCE_AVG);
	imagePointsMean = cv::repeat(imagePointsMean, matImagePoints.rows, 1); // get T_13 and T_23 from imagePointsMean
	matImagePoints = matImagePoints - imagePointsMean;
	// scale the image points such that the RMS distance from the origin is sqrt(2):
	// 1) calculate the average norm (root mean squared distance) of all vectors
	float averageNorm = 0.0f;
	for (int row = 0; row < matImagePoints.rows; ++row) {
		averageNorm += cv::norm(matImagePoints.row(row), cv::NORM_L2);
	}
	averageNorm /= matImagePoints.rows;
	// 2) multiply every vectors coordinate by sqrt(2)/avgnorm
	float scaleFactor = std::sqrt(2)/averageNorm;
	matImagePoints *= scaleFactor; // add unit homogeneous component here
	// The points in matImagePoints now have a RMS distance from the origin of sqrt(2).
	// The normalisation matrix so that the 2D points are mean-free and their norm is as described above.
	Mat T = Mat::zeros(3, 3, CV_32FC1);
	T.at<float>(0, 0) = scaleFactor; // s_x
	T.at<float>(1, 1) = scaleFactor; // s_y
	T.at<float>(0, 2) = -imagePointsMean.at<float>(0, 0) * scaleFactor; // t_x
	T.at<float>(1, 2) = -imagePointsMean.at<float>(0, 1) * scaleFactor; // t_y
	T.at<float>(2, 2) = 1;
	
	// center the model points to the origin:
	Mat tmpOrigMdlPoints = matModelPoints.clone(); // Temp for testing: Save the original coordinates.
	// translate the centroid of the model points to the origin:
	Mat modelPointsMean; // use non-homogeneous coords for the next few steps? (less submatrices etc overhead)
	cv::reduce(matModelPoints, modelPointsMean, 0, CV_REDUCE_AVG);
	modelPointsMean = cv::repeat(modelPointsMean, matModelPoints.rows, 1);
	matModelPoints = matModelPoints - modelPointsMean;
	// scale the model points such that the RMS distance from the origin is sqrt(3):
	// 1) calculate the average norm (root mean squared distance) of all vectors
	averageNorm = 0.0f;
	for (int row = 0; row < matModelPoints.rows; ++row) {
		averageNorm += cv::norm(matModelPoints.row(row), cv::NORM_L2);
	}
	averageNorm /= matModelPoints.rows;
	// 2) multiply every vectors coordinate by sqrt(3)/avgnorm
	scaleFactor = std::sqrt(3) / averageNorm;
	matModelPoints *= scaleFactor; // add unit homogeneous component here
	// The points in matModelPoints now have a RMS distance from the origin of sqrt(3).
	// The normalisation matrix so that the 3D points are mean-free and their norm is as described above.
	Mat U = Mat::zeros(4, 4, CV_32FC1);
	U.at<float>(0, 0) = scaleFactor; // s_x
	U.at<float>(1, 1) = scaleFactor; // s_y
	U.at<float>(2, 2) = scaleFactor; // s_z
	U.at<float>(0, 3) = -modelPointsMean.at<float>(0, 0) * scaleFactor; // t_x
	U.at<float>(1, 3) = -modelPointsMean.at<float>(0, 1) * scaleFactor; // t_y
	U.at<float>(2, 3) = -modelPointsMean.at<float>(0, 2) * scaleFactor; // t_z
	U.at<float>(3, 3) = 1;

	// Estimate the normalized camera matrix (C tilde).
	// We are solving the system $A_8 * p_8 = b$
	// The solution is obtained by the pseudo-inverse of A_8:
	// $p_8 = A_8^+ * b$
	Mat A_8 = Mat::zeros(numCorrespondences * 2, 8, CV_32FC1);
	//Mat p_8(); // p_8 is 8 x 1. We are solving for it.
	Mat b(numCorrespondences * 2, 1, CV_32FC1);
	for (int i = 0; i < numCorrespondences; ++i) {
		A_8.at<float>(2*i, 0) = matModelPoints.at<float>(i, 0); // could maybe made faster by assigning the whole row/col-range if possible?
		A_8.at<float>(2*i, 1) = matModelPoints.at<float>(i, 1);
		A_8.at<float>(2*i, 2) = matModelPoints.at<float>(i, 2);
		A_8.at<float>(2*i, 3) = 1;
		A_8.at<float>((2*i)+1, 4) = matModelPoints.at<float>(i, 0);
		A_8.at<float>((2*i)+1, 5) = matModelPoints.at<float>(i, 1);
		A_8.at<float>((2*i)+1, 6) = matModelPoints.at<float>(i, 2);
		A_8.at<float>((2*i)+1, 7) = 1;
		b.at<float>(2*i, 0) = matImagePoints.at<float>(i, 0);
		b.at<float>((2*i)+1, 0) = matImagePoints.at<float>(i, 1);
	}
	Mat p_8 = A_8.inv(cv::DECOMP_SVD) * b;
	Mat C_tilde = Mat::zeros(3, 4, CV_32FC1);
	C_tilde.row(0) = p_8.rowRange(0, 4).t(); // The first row of C_tilde consists of the first 4 entries of the col-vector p_8
	C_tilde.row(1) = p_8.rowRange(4, 8).t(); // Second row = last 4 entries
	C_tilde.at<float>(2, 3) = 1; // the last row is [0 0 0 1]

	Mat P_Affine = T.inv() * C_tilde * U;
	return P_Affine;
}

Mat calculateAffineZDirection(Mat affineCameraMatrix)
{
	//affineCameraMatrix is the original 3x4 affine matrix. But we return a 4x4 matrix with a z-rotation (viewing direction) as well(for the z-buffering)
	// Take the cross product of row 0 with row 1 to get the direction perpendicular to the viewing plane (= the viewing direction).
	// Todo: We should check if we look/project into the right direction - the sign could be wrong?
	Mat affineCamZ = affineCameraMatrix.row(0).colRange(0, 3).cross(affineCameraMatrix.row(1).colRange(0, 3));
	affineCamZ /= cv::norm(affineCamZ, cv::NORM_L2);

	// The 4x4 affine camera matrix
	Mat affineCamFull = Mat::zeros(4, 4, CV_32FC1);

	// Replace the third row with the camera-direction (z)
	Mat affineCamSubMat = affineCamFull.row(2).colRange(0, 3);
	affineCamZ.copyTo(affineCamSubMat); // Set first 3 components. 4th component stays 0.
	
	// Copy the first 2 rows from the input matrix
	Mat affineCamFullSub = affineCamFull.rowRange(0, 2);
	affineCameraMatrix.rowRange(0, 2).copyTo(affineCamFullSub); 
	
	// The 4th row is (0, 0, 0, 1):
	affineCamFull.at<float>(3, 3) = 1.0f; 

	return affineCamFull;
}

cv::Vec2f projectAffine(cv::Vec3f vertex, cv::Mat affineCameraMatrix, int screenWidth, int screenHeight)
{
	Mat vertex4(4, 1, CV_32FC1);
	vertex4.at<float>(0, 0) = vertex[0];
	vertex4.at<float>(1, 0) = vertex[1];
	vertex4.at<float>(2, 0) = vertex[2];
	vertex4.at<float>(3, 0) = 1;
	// Transform to clip space:
	Mat clipCoords = affineCameraMatrix * vertex4;
	// Take the x and y coordinates in clip space and apply the window transform:
	cv::Vec2f screenCoords = render::utils::clipToScreenSpace(cv::Vec2f(clipCoords.rowRange(0, 2)), screenWidth, screenHeight);
	return screenCoords;
}

} /* namespace fitting */
