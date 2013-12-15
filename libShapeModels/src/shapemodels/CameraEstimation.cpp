/*
 * CameraEstimation.cpp
 *
 *  Created on: 15.12.2013
 *      Author: Patrik Huber
 */

#include "shapemodels/CameraEstimation.hpp"

#include "logging/LoggerFactory.hpp"

#include "opencv2/calib3d/calib3d.hpp"
/*
#include "boost/lexical_cast.hpp"
#include <exception>

using cv::Vec3f;
using boost::lexical_cast;
using std::string;
*/
using logging::LoggerFactory;
using cv::Mat;
using std::vector;
using std::pair;

namespace shapemodels {

CameraEstimation::CameraEstimation(/* const? shared_ptr? */MorphableModel morphableModel) : morphableModel(morphableModel)
{

}

std::pair<cv::Mat, cv::Mat> CameraEstimation::estimate(std::vector<imageio::ModelLandmark> imagePoints, cv::Mat intrinsicCameraMatrix, std::vector<int> vertexIds /*= std::vector<int>()*/)
{
	if (imagePoints.size() < 3) {
		Loggers->getLogger("shapemodels").error("CameraEstimation: Number of points given is smaller than 3.");
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
	return std::make_pair(rvec, tvec);
}

}