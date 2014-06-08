/*
 * OpenCVCameraEstimation.cpp
 *
 *  Created on: 15.12.2013
 *      Author: Patrik Huber
 */

#include "fitting/OpenCVCameraEstimation.hpp"

#include "logging/LoggerFactory.hpp"

#include "opencv2/calib3d/calib3d.hpp"

using logging::LoggerFactory;
using morphablemodel::MorphableModel;
using cv::Mat;
using cv::Point2f;
using cv::Point3f;
using std::vector;
using std::pair;

namespace fitting {

OpenCVCameraEstimation::OpenCVCameraEstimation(/* const? shared_ptr? */MorphableModel morphableModel) : morphableModel(morphableModel)
{

}

cv::Mat OpenCVCameraEstimation::estimate(std::vector<imageio::ModelLandmark> imagePoints, cv::Mat intrinsicCameraMatrix, std::vector<int> vertexIds /*= std::vector<int>()*/)
{
	if (imagePoints.size() < 3) {
		Loggers->getLogger("morphablemodel").error("CameraEstimation: Number of points given is smaller than 3.");
		throw std::runtime_error("CameraEstimation: Number of points given is smaller than 3.");
	}

	// Todo: Currently, the optional vertexIds is not used
	vector<Point2f> points2d;
	vector<Point3f> points3d;
	for (const auto& landmark : imagePoints) {
		points2d.emplace_back(landmark.getPoint2D());
		points3d.emplace_back(morphableModel.getShapeModel().getMeanAtPoint(landmark.getName()));
	}

	//Estimate the pose
	Mat rvec(3, 1, CV_64FC1);
	Mat tvec(3, 1, CV_64FC1);
	if (points2d.size() == 3) {
		cv::solvePnP(points3d, points2d, intrinsicCameraMatrix, vector<float>(), rvec, tvec, false, CV_ITERATIVE); // CV_ITERATIVE (3pts) | CV_P3P (4pts) | CV_EPNP (4pts)
	} else {
		cv::solvePnP(points3d, points2d, intrinsicCameraMatrix, vector<float>(), rvec, tvec, false, CV_EPNP); // CV_ITERATIVE (3pts) | CV_P3P (4pts) | CV_EPNP (4pts)
		// Alternative, more outlier-resistant:
		// cv::solvePnPRansac(modelPoints, imagePoints, camMatrix, distortion, rvec, tvec, false); // min 4 points
		// has an optional argument 'inliers' - might be useful
	}

	// Convert rvec/tvec to matrices, etc... return 4x4 extrinsic camera matrix
	Mat rotation_matrix(3, 3, CV_64FC1);
	cv::Rodrigues(rvec, rotation_matrix);
	rotation_matrix.convertTo(rotation_matrix, CV_32FC1);
	Mat translation_vector = tvec;
	translation_vector.convertTo(translation_vector, CV_32FC1);

	Mat extrinsicCameraMatrix = Mat::zeros(4, 4, CV_32FC1);
	Mat extrRot = extrinsicCameraMatrix(cv::Range(0, 3), cv::Range(0, 3));
	rotation_matrix.copyTo(extrRot);
	Mat extrTrans = extrinsicCameraMatrix(cv::Range(0, 3), cv::Range(3, 4));
	translation_vector.copyTo(extrTrans);
	extrinsicCameraMatrix.at<float>(3, 3) = 1; // maybe set (3, 2) = 1 here instead so that the renderer can do divByW as well? (see Todo in libRender)

	return extrinsicCameraMatrix;
}

cv::Mat OpenCVCameraEstimation::createIntrinsicCameraMatrix(float f, int w, int h)
{
	Mat camMatrix = (cv::Mat_<double>(3, 3) << f, 0, w / 2.0,
											   0, f, h / 2.0,
											   0, 0, 1.0    );
	return camMatrix;
}

} /* namespace fitting */
