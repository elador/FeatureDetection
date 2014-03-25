/*
 * AffineCameraEstimation.cpp
 *
 *  Created on: 31.12.2013
 *      Author: Patrik Huber
 */

#include "morphablemodel/AffineCameraEstimation.hpp"

#include "logging/LoggerFactory.hpp"

#include "opencv2/core/core_c.h"
/*
#include "boost/lexical_cast.hpp"
#include <exception>

using cv::Vec3f;
using boost::lexical_cast;
using std::string;
*/
using logging::LoggerFactory;
using cv::Mat;
using cv::Vec3f;
using cv::Vec4f;
using std::vector;

namespace shapemodels {

AffineCameraEstimation::AffineCameraEstimation(/* const? shared_ptr? */MorphableModel morphableModel) : morphableModel(morphableModel)
{

}

cv::Mat AffineCameraEstimation::estimate(std::vector<imageio::ModelLandmark> imagePoints, std::vector<int> vertexIds /*= std::vector<int>()*/)
{
	if (imagePoints.size() < 4) {
		Loggers->getLogger("shapemodels").error("AffineCameraEstimation: Number of points given needs to be equal to or larger than 4.");
		throw std::runtime_error("AffineCameraEstimation: Number of points given needs to be equal to or larger than 4.");
	}

	// Todo: Currently, the optional vertexIds is not used
	Mat matImagePoints(imagePoints.size(), 2, CV_32FC1);
	Mat matModelPoints(imagePoints.size(), 3, CV_32FC1);
	int row = 0;
	for (const auto& landmark : imagePoints) {
		Vec3f tmp = morphableModel.getShapeModel().getMeanAtPoint(landmark.getName());
		matImagePoints.at<float>(row, 0) = landmark.getPoint2D().x;
		matImagePoints.at<float>(row, 1) = landmark.getPoint2D().y;
		matModelPoints.at<float>(row, 0) = tmp[0];
		matModelPoints.at<float>(row, 1) = tmp[1];
		matModelPoints.at<float>(row, 2) = tmp[2];
		++row;
	}
	Mat tmpOrigImgPoints = matImagePoints.clone(); // temp for testing
	// translate the centroid of the image points to the origin:
	Mat imagePointsMean; // use non-homogeneous coords for the next few steps? (less submatrices etc overhead)
	cv::reduce(matImagePoints, imagePointsMean, 0, CV_REDUCE_AVG);
	imagePointsMean = cv::repeat(imagePointsMean, imagePoints.size(), 1); // get T_13 and T_23 from imagePointsMean
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
	Mat T = Mat::zeros(3, 3, CV_32FC1);
	T.at<float>(0, 0) = scaleFactor; // s_x
	T.at<float>(1, 1) = scaleFactor; // s_y
	T.at<float>(0, 2) = -imagePointsMean.at<float>(0, 0) * scaleFactor; // t_x
	T.at<float>(1, 2) = -imagePointsMean.at<float>(0, 1) * scaleFactor; // t_y
	T.at<float>(2, 2) = 1;

	Vec3f testPoint;
	testPoint[0] = tmpOrigImgPoints.row(0).at<float>(0);
	testPoint[1] = tmpOrigImgPoints.row(0).at<float>(1);
	testPoint[2] = 1;
	Mat testPointM(testPoint);
	Mat res = T * testPointM;

	// center the model points to the origin:
	Mat tmpOrigMdlPoints = matModelPoints.clone(); // temp for testing
	// translate the centroid of the model points to the origin:
	Mat modelPointsMean; // use non-homogeneous coords for the next few steps? (less submatrices etc overhead)
	cv::reduce(matModelPoints, modelPointsMean, 0, CV_REDUCE_AVG);
	modelPointsMean = cv::repeat(modelPointsMean, imagePoints.size(), 1);
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
	Mat U = Mat::zeros(4, 4, CV_32FC1);
	U.at<float>(0, 0) = scaleFactor; // s_x
	U.at<float>(1, 1) = scaleFactor; // s_y
	U.at<float>(2, 2) = scaleFactor; // s_z
	U.at<float>(0, 3) = -modelPointsMean.at<float>(0, 0) * scaleFactor; // t_x
	U.at<float>(1, 3) = -modelPointsMean.at<float>(0, 1) * scaleFactor; // t_y
	U.at<float>(2, 3) = -modelPointsMean.at<float>(0, 2) * scaleFactor; // t_z
	U.at<float>(3, 3) = 1;

	Vec4f testPoint3d;
	testPoint3d[0] = tmpOrigMdlPoints.row(6).at<float>(0);
	testPoint3d[1] = tmpOrigMdlPoints.row(6).at<float>(1);
	testPoint3d[2] = tmpOrigMdlPoints.row(6).at<float>(2);
	testPoint3d[3] = 1;
	Mat testPointM3d(testPoint3d);
	Mat res3d = U * testPointM3d;

	// Estimate the normalized camera matrix (C tilde).
	// We are solving the system $A_8 * p_8 = b$
	// The solution is obtained by the pseudo-inverse of A_8:
	// $p_8 = A_8^+ * b$
	Mat A_8 = Mat::zeros(imagePoints.size()*2, 8, CV_32FC1);
	//Mat p_8(); // p_8 is 8 x 1. We are solving for it.
	Mat b(imagePoints.size()*2, 1, CV_32FC1);
	for (int i = 0; i < imagePoints.size(); ++i) {
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
	C_tilde.at<float>(0, 0) = p_8.at<float>(0, 0); // could maybe made faster by assigning the whole row/col-range if possible?
	C_tilde.at<float>(0, 1) = p_8.at<float>(1, 0);
	C_tilde.at<float>(0, 2) = p_8.at<float>(2, 0);
	C_tilde.at<float>(0, 3) = p_8.at<float>(3, 0);
	C_tilde.at<float>(1, 0) = p_8.at<float>(4, 0);
	C_tilde.at<float>(1, 1) = p_8.at<float>(5, 0);
	C_tilde.at<float>(1, 2) = p_8.at<float>(6, 0);
	C_tilde.at<float>(1, 3) = p_8.at<float>(7, 0);
	C_tilde.at<float>(2, 3) = 1;

	Mat P_Affine = T.inv() * C_tilde * U;

	Mat restest = P_Affine * testPointM3d;

	return P_Affine;
}

cv::Mat AffineCameraEstimation::calculateFullMatrix(cv::Mat affineCameraMatrix)
{
	//affineCameraMatrix is the original 3x4 affine matrix. But we return a 4x4 matrix with a z - rotation(viewing - direction) as well(for the z - buffering)
	/* affineCam = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 0, 1); */
	Mat affineCamZ = affineCameraMatrix.row(0).colRange(0, 3).cross(affineCameraMatrix.row(1).colRange(0, 3));
	affineCamZ /= cv::norm(affineCamZ, cv::NORM_L2);

	// Replace the third row with the camera-direction (z)
	// Todo: Take care of sign
	Mat affineCamSubMat = affineCameraMatrix.row(2).colRange(0, 3);
	affineCamZ.copyTo(affineCamSubMat);
	affineCameraMatrix.at<float>(2, 3) = 0;

	Mat affineCamFull = Mat::zeros(4, 4, CV_32FC1);
	Mat affineCamFullSub = affineCamFull.rowRange(0, 2);
	affineCameraMatrix.rowRange(0, 2).copyTo(affineCamFullSub);
	affineCamFullSub = affineCamFull.row(2).colRange(0, 3);
	affineCamZ.copyTo(affineCamFullSub);
	affineCamFull.at<float>(2, 3) = 0.0f;
	affineCamFull.at<float>(3, 3) = 1.0f; // 4th row is (0, 0, 0, 1)

	return affineCamFull;
}

}