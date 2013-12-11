/*
 * FeaturePointsEvaluator.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef FEATUREPOINTSEVALUATOR_HPP_
#define FEATUREPOINTSEVALUATOR_HPP_

#include "shapemodels/MorphableModel.hpp"

#include "imageprocessing/Patch.hpp"
#include "render/RenderDevicePnP.hpp"
#include "render/Camera.hpp"
#include "render/MatrixUtils.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/legacy/compat.hpp"

#include <memory>
#include <iostream>
#include <utility>

using cv::Point2f;
using cv::Point3f;

using std::unique_ptr;
using std::pair;
using std::map;
using std::string;
using std::vector;

namespace shapemodels {

/**
 * A ... .
 * Note for POSIT: I think it would be better to use 3 points and a simple "rigid projection" of
 * the model (as in Sami's paper). Then we could also do a fast reject, and have a proper RANSAC
 * around it.
 * Note2: The first point is always used as camera center I think. Try using 3 points and
 * the center of those 3 in the model as a 4th (first) ?
 *
 * Note 3: Try render it with R=id, t=0. Points flipped. Bc OCV image origin is up-left. Look
 *         in our renderer code, cam-matrix y has to be flipped.
 *
 * Problem with POSIT. 3 solutions:
 *  - use solvePnP (does only estimate R and T, not the cam-matrix (e.g. focal length)
 *  - try with 3 points + their center
 *  - analyze the problem, draw the model, ...
 */
class FeaturePointsEvaluator {
public:
	FeaturePointsEvaluator() {}; // delete this
	FeaturePointsEvaluator(MorphableModel mm) : mm(mm) {};
	~FeaturePointsEvaluator() {};

	/**
	 * Does xyz
	 *
	 * @param[in] param The parameter.
	 */
	//virtual void func(int param);

	/**
	 * A getter.
	 *
	 * @return Something.
	 */
	//virtual const int getter() const;

	std::pair<Mat, Mat> doPosit(map<string, std::shared_ptr<imageprocessing::Patch>> landmarkPoints, Mat img) { // img for debug purposes

		// Convert patch to Point2f
		map<string, Point2f> landmarkPointsAsPoint2f;
		for (const auto& p : landmarkPoints) {
			landmarkPointsAsPoint2f.insert(make_pair(p.first, Point2f(p.second->getX(), p.second->getY())));
		}

		map<string, Point3f> mmVertices = get3dmmLmsFromFfps(landmarkPointsAsPoint2f, mm);
		originVertex = begin(mmVertices)->second;
		mmVertices = movePoints(mmVertices, -begin(mmVertices)->second); // move all points so that the first one is now (0, 0, 0)

		//Create the model points
		vector<CvPoint3D32f> modelPoints;

		for (const auto& v : mmVertices) {
			modelPoints.push_back(cvPoint3D32f(v.second.x, v.second.y, v.second.z));
		}

		//Create the image points 
		vector<CvPoint2D32f> srcImagePoints;
		for (const auto& p : landmarkPointsAsPoint2f) {
			Point2f positCoords = imgToPositCoords(p.second, img.cols, img.rows);
			srcImagePoints.push_back(cvPoint2D32f(positCoords.x, positCoords.y));
		}

		//Create the POSIT object with the model points
		unique_ptr<CvPOSITObject, void(*)(CvPOSITObject*)> positObject = unique_ptr<CvPOSITObject, void(*)(CvPOSITObject*)>(cvCreatePOSITObject(&modelPoints[0], (int)modelPoints.size()), [](CvPOSITObject *p) { cvReleasePOSITObject(&p); });
		
		//Estimate the pose
		Mat rotation_matrix(3, 3, CV_32FC1); // makes sure that isContinuous()==true
		Mat translation_vector(1, 3, CV_32FC1); // 1 row, 3 columns
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1.0e-4f);
		cvPOSIT(positObject.get(), &srcImagePoints[0], 1000, criteria, rotation_matrix.ptr<float>(0), translation_vector.ptr<float>(0)); // 1000 = focal length

		// Visualize the image
		// - the 4 selected 2D landmarks
		// - the 4 3D 3dmm landmarks projected with T and R to 2D
		// - evtl overlay the whole model

		for (const auto& p : landmarkPoints) {
			cv::rectangle(img, cv::Point(cvRound(p.second->getX()-2.0f), cvRound(p.second->getY()-2.0f)), cv::Point(cvRound(p.second->getX()+2.0f), cvRound(p.second->getY()+2.0f)), cv::Scalar(255, 0, 0));
			drawFfpsText(img, make_pair(p.first, Point2f(p.second->getX(), p.second->getY())));
		}
		for (const auto& v : mmVertices) {
			Point2f projpoint = positProject(v.second, translation_vector, rotation_matrix, 1000.0f);
			projpoint = positToImgCoords(projpoint, img.cols, img.rows);

			drawFfpsCircle(img, make_pair(v.first, projpoint));
			drawFfpsText(img, make_pair(v.first, projpoint));
		}
			
		return std::make_pair(translation_vector, rotation_matrix);
	};

	float evaluateNewPoint(string landmarkName, std::shared_ptr<imageprocessing::Patch> patch, Mat trans, Mat rot, Mat img) {
		// Get the new vertex from 3DMM, project into 2D with trans and rot matrix calculated by cvPOSIT
		// Convert patch to Point2f
		map<string, Point2f> landmarkPointAsPoint2f;
		landmarkPointAsPoint2f.insert(make_pair(landmarkName, Point2f(patch->getX(), patch->getY())));

		// Get the vertex from the 3dmm
		map<string, Point3f> vertex = get3dmmLmsFromFfps(landmarkPointAsPoint2f, mm);
		// move the vertex to correspond to the right origin
		vertex = movePoints(vertex, -originVertex);

		// calculate the distance between the projected point and the given landmark
		Point2f projPoint = positProject((*begin(vertex)).second, trans, rot, 1000.0f);
		projPoint = positToImgCoords(projPoint, img.cols, img.rows);

		// draw the original landmark
		cv::rectangle(img, cv::Point(cvRound(patch->getX()-2.0f), cvRound(patch->getY()-2.0f)), cv::Point(cvRound(patch->getX()+2.0f), cvRound(patch->getY()+2.0f)), cv::Scalar(255, 0, 0));
		drawFfpsText(img, make_pair(landmarkName, Point2f(patch->getX(), patch->getY())));
		// draw the projected vertex
		drawFfpsCircle(img, make_pair(landmarkName, projPoint));
		drawFfpsText(img, make_pair(landmarkName, projPoint));

		float distance = 0.0f;
		distance = cv::norm(projPoint - (*begin(landmarkPointAsPoint2f)).second);
		return distance;
	};


	std::pair<Mat, Mat> doPnP(map<string, std::shared_ptr<imageprocessing::Patch>> landmarkPoints, Mat img) { // img for debug purposes

		// Convert patch to Point2f
		map<string, Point2f> landmarkPointsAsPoint2f;
		vector<Point2f> imagePoints;
		for (const auto& p : landmarkPoints) {
			landmarkPointsAsPoint2f.insert(make_pair(p.first, Point2f(p.second->getX(), p.second->getY())));
			imagePoints.emplace_back(Point2f(p.second->getX(), p.second->getY()));
		}

		// Get and create the model points
		map<string, Point3f> mmVertices = get3dmmLmsFromFfps(landmarkPointsAsPoint2f, mm);
		vector<Point3f> modelPoints;
		for (const auto& v : mmVertices) {
			modelPoints.push_back(Point3f(v.second.x, v.second.y, v.second.z));
		}

		//Estimate the pose
		int max_d = std::max(img.rows,img.cols); // should be the focal length? (don't forget the aspect ratio!). TODO Read in Hartley-Zisserman what this is
		Mat camMatrix = (cv::Mat_<double>(3,3) << max_d, 0,		img.cols/2.0,
												  0,	 max_d, img.rows/2.0,
												  0,	 0,		1.0);
		Mat rvec(3, 1, CV_64FC1);
		Mat tvec(3, 1, CV_64FC1);
		if (imagePoints.size() == 3) {
			solvePnP(modelPoints, imagePoints, camMatrix, vector<float>(), rvec, tvec, false, CV_ITERATIVE); // CV_ITERATIVE (3pts) | CV_P3P (4pts) | CV_EPNP (4pts)
		} else {
			solvePnP(modelPoints, imagePoints, camMatrix, vector<float>(), rvec, tvec, false, CV_EPNP); // CV_ITERATIVE (3pts) | CV_P3P (4pts) | CV_EPNP (4pts)
		}
		//solvePnPRansac(modelPoints, imagePoints, camMatrix, distortion, rvec, tvec, false); // min 4 points

		Mat rotation_matrix(3, 3, CV_64FC1);
		Rodrigues(rvec, rotation_matrix);
		rotation_matrix.convertTo(rotation_matrix, CV_32FC1);
		Mat translation_vector = tvec;
		translation_vector.convertTo(translation_vector, CV_32FC1);

		camMatrix.convertTo(camMatrix, CV_32FC1);

		cameraMatrix = camMatrix;
		rodrRotVec = rvec;
		rodrTransVec = tvec;

		// Visualize the image
		// - the 4 selected 2D landmarks
		// - the 4 3D 3dmm landmarks projected with T and R to 2D
		// - evtl overlay the whole model

		for (const auto& p : landmarkPoints) {
			cv::rectangle(img, cv::Point(cvRound(p.second->getX()-2.0f), cvRound(p.second->getY()-2.0f)), cv::Point(cvRound(p.second->getX()+2.0f), cvRound(p.second->getY()+2.0f)), cv::Scalar(255, 0, 0));
			drawFfpsText(img, make_pair(p.first, Point2f(p.second->getX(), p.second->getY())));
		}
		//vector<Point2f> projectedPoints;
		//projectPoints(modelPoints, rvec, tvec, camMatrix, vector<float>(), projectedPoints); // same result as below
		Mat extrinsicCameraMatrix = Mat::zeros(4, 4, CV_32FC1);
		Mat extrRot = extrinsicCameraMatrix(cv::Range(0, 3), cv::Range(0, 3));
		rotation_matrix.copyTo(extrRot);
		Mat extrTrans = extrinsicCameraMatrix(cv::Range(0, 3), cv::Range(3, 4));
		translation_vector.copyTo(extrTrans);
		extrinsicCameraMatrix.at<float>(3, 3) = 1;

		Mat intrinsicCameraMatrix = Mat::zeros(4, 4, CV_32FC1);
		Mat intrinsicCameraMatrixMain = intrinsicCameraMatrix(cv::Range(0, 3), cv::Range(0, 3));
		camMatrix.copyTo(intrinsicCameraMatrixMain);
		intrinsicCameraMatrix.at<float>(3, 3) = 1;

		for (const auto& v : mmVertices) {
			Mat vertex(v.second);
			Mat vertex_homo = Mat::ones(4, 1, CV_32FC1);
			Mat vertex_homo_coords = vertex_homo(cv::Range(0, 3), cv::Range(0, 1));
			vertex.copyTo(vertex_homo_coords);
			Mat v2 = rotation_matrix * vertex;
			Mat v3 = v2 + translation_vector;
			Mat v3_mat = extrinsicCameraMatrix * vertex_homo;

			Mat v4 = camMatrix * v3;
			Mat v4_mat = intrinsicCameraMatrix * v3_mat;

			Point3f v4p(v4);
			Point2f v4p2d(v4p.x/v4p.z, v4p.y/v4p.z); // if != 0
			Point3f v4p_homo(v4_mat(cv::Range(0, 3), cv::Range(0, 1)));
			Point2f v4p2d_homo(v4p_homo.x/v4p_homo.z, v4p_homo.y/v4p_homo.z); // if != 0
			drawFfpsCircle(img, make_pair(v.first, v4p2d_homo));
			drawFfpsText(img, make_pair(v.first, v4p2d_homo));
		}

		//shapemodels::MorphableModel mm = shapemodels::MorphableModel::loadScmModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\SurreyLowResGuosheng\\NON3448\\ShpVtxModelBin_NON3448.scm", "C:\\Users\\Patrik\\Documents\\GitHub\\featurePoints_SurreyScm.txt");
		std::shared_ptr<render::Mesh> meshToDraw = std::make_shared<render::Mesh>(mm.getMean());
		
		const float aspect = (float)img.cols/(float)img.rows; // 640/480
		render::Camera camera(Vec3f(0.0f, 0.0f, 0.0f), /*horizontalAngle*/0.0f*(CV_PI/180.0f), /*verticalAngle*/0.0f*(CV_PI/180.0f), render::Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, /*zNear*/-0.1f, /*zFar*/-100.0f));
		render::RenderDevicePnP r(img.cols, img.rows, camera); // 640, 480
		//r.setModelTransform(render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f));
		r.setIntrinsicCameraTransform(intrinsicCameraMatrix);
		r.setExtrinsicCameraTransform(extrinsicCameraMatrix);
		r.draw(meshToDraw, nullptr);
		Mat buff = r.getImage();
		Mat buffWithoutAlpha;
		//buff.convertTo(buffWithoutAlpha, CV_BGRA2BGR);
		cvtColor(buff, buffWithoutAlpha, cv::COLOR_BGRA2BGR);
		Mat weighted = img.clone(); // get the right size
		cv::addWeighted(img, 0.4, buffWithoutAlpha, 0.6, 0.0, weighted);
		return std::make_pair(translation_vector, rotation_matrix);
	};

	float evaluateNewPointPnP(string landmarkName, std::shared_ptr<imageprocessing::Patch> patch, Mat trans, Mat rot, Mat img) {
		// Get the new vertex from 3DMM, project into 2D with trans and rot matrix calculated by cvPOSIT
		// Convert patch to Point2f
		map<string, Point2f> landmarkPointAsPoint2f;
		landmarkPointAsPoint2f.insert(make_pair(landmarkName, Point2f(patch->getX(), patch->getY())));

		// Get the vertex from the 3dmm
		map<string, Point3f> vertex = get3dmmLmsFromFfps(landmarkPointAsPoint2f, mm);
		vector<Point3f> modelPoint;
		modelPoint.emplace_back((*begin(vertex)).second);

		// calculate the distance between the projected point and the given landmark
		vector<Point2f> projectedPoints;
		projectPoints(modelPoint, rodrRotVec, rodrTransVec, cameraMatrix, vector<float>(), projectedPoints); // same result as below
		Point2f projPoint = projectedPoints[0];

		// draw the original landmark
		cv::rectangle(img, cv::Point(cvRound(patch->getX()-2.0f), cvRound(patch->getY()-2.0f)), cv::Point(cvRound(patch->getX()+2.0f), cvRound(patch->getY()+2.0f)), cv::Scalar(255, 0, 0));
		drawFfpsText(img, make_pair(landmarkName, Point2f(patch->getX(), patch->getY())));
		// draw the projected vertex
		drawFfpsCircle(img, make_pair(landmarkName, projPoint));
		drawFfpsText(img, make_pair(landmarkName, projPoint));

		float distance = 0.0f;
		distance = cv::norm(projPoint - (*begin(landmarkPointAsPoint2f)).second);
		return distance;
	};

	cv::Rect getFaceCenter(map<string, shared_ptr<imageprocessing::Patch>> landmarkPoints, Mat img) { // img shouldn't be necessary
		// Convert patch to Point2f
		map<string, Point2f> landmarkPointsAsPoint2f;
		vector<Point2f> imagePoints;
		for (const auto& p : landmarkPoints) {
			landmarkPointsAsPoint2f.insert(make_pair(p.first, Point2f(p.second->getX(), p.second->getY())));
			imagePoints.emplace_back(Point2f(p.second->getX(), p.second->getY()));
		}

		// Get and create the model points
		map<string, Point3f> mmVertices = get3dmmLmsFromFfps(landmarkPointsAsPoint2f, mm);
		vector<Point3f> modelPoints;
		for (const auto& v : mmVertices) {
			modelPoints.push_back(Point3f(v.second.x, v.second.y, v.second.z));
		}

		//Estimate the pose
		int max_d = std::max(img.rows,img.cols); // should be the focal length? (don't forget the aspect ratio!). TODO Read in Hartley-Zisserman what this is
		Mat camMatrix = (cv::Mat_<double>(3,3) << max_d, 0,		img.cols/2.0,
			0,	 max_d, img.rows/2.0,
			0,	 0,		1.0);
		Mat rvec(3, 1, CV_64FC1);
		Mat tvec(3, 1, CV_64FC1);
		if (imagePoints.size() == 3) {
			solvePnP(modelPoints, imagePoints, camMatrix, vector<float>(), rvec, tvec, false, CV_ITERATIVE); // CV_ITERATIVE (3pts) | CV_P3P (4pts) | CV_EPNP (4pts)
		} else {
			solvePnP(modelPoints, imagePoints, camMatrix, vector<float>(), rvec, tvec, false, CV_EPNP); // CV_ITERATIVE (3pts) | CV_P3P (4pts) | CV_EPNP (4pts)
		}
		//solvePnPRansac(modelPoints, imagePoints, camMatrix, distortion, rvec, tvec, false); // min 4 points

		Mat rotation_matrix(3, 3, CV_64FC1);
		Rodrigues(rvec, rotation_matrix);
		rotation_matrix.convertTo(rotation_matrix, CV_32FC1);
		Mat translation_vector = tvec;
		translation_vector.convertTo(translation_vector, CV_32FC1);

		camMatrix.convertTo(camMatrix, CV_32FC1);

		cameraMatrix = camMatrix;
		rodrRotVec = rvec;
		rodrTransVec = tvec;

		// Visualize the image
		// - the 4 selected 2D landmarks
		// - the 4 3D 3dmm landmarks projected with T and R to 2D
		// - evtl overlay the whole model

		for (const auto& p : landmarkPoints) {
			cv::rectangle(img, cv::Point(cvRound(p.second->getX()-2.0f), cvRound(p.second->getY()-2.0f)), cv::Point(cvRound(p.second->getX()+2.0f), cvRound(p.second->getY()+2.0f)), cv::Scalar(255, 0, 0));
			drawFfpsText(img, make_pair(p.first, Point2f(p.second->getX(), p.second->getY())));
		}
		//vector<Point2f> projectedPoints;
		//projectPoints(modelPoints, rvec, tvec, camMatrix, vector<float>(), projectedPoints); // same result as below
		Mat extrinsicCameraMatrix = Mat::zeros(4, 4, CV_32FC1);
		Mat extrRot = extrinsicCameraMatrix(cv::Range(0, 3), cv::Range(0, 3));
		rotation_matrix.copyTo(extrRot);
		Mat extrTrans = extrinsicCameraMatrix(cv::Range(0, 3), cv::Range(3, 4));
		translation_vector.copyTo(extrTrans);
		extrinsicCameraMatrix.at<float>(3, 3) = 1;

		Mat intrinsicCameraMatrix = Mat::zeros(4, 4, CV_32FC1);
		Mat intrinsicCameraMatrixMain = intrinsicCameraMatrix(cv::Range(0, 3), cv::Range(0, 3));
		camMatrix.copyTo(intrinsicCameraMatrixMain);
		intrinsicCameraMatrix.at<float>(3, 3) = 1;

		for (const auto& v : mmVertices) {
			Mat vertex(v.second);
			Mat vertex_homo = Mat::ones(4, 1, CV_32FC1);
			Mat vertex_homo_coords = vertex_homo(cv::Range(0, 3), cv::Range(0, 1));
			vertex.copyTo(vertex_homo_coords);
			Mat v2 = rotation_matrix * vertex;
			Mat v3 = v2 + translation_vector;
			Mat v3_mat = extrinsicCameraMatrix * vertex_homo;

			Mat v4 = camMatrix * v3;
			Mat v4_mat = intrinsicCameraMatrix * v3_mat;

			Point3f v4p(v4);
			Point2f v4p2d(v4p.x/v4p.z, v4p.y/v4p.z); // if != 0
			Point3f v4p_homo(v4_mat(cv::Range(0, 3), cv::Range(0, 1)));
			Point2f v4p2d_homo(v4p_homo.x/v4p_homo.z, v4p_homo.y/v4p_homo.z); // if != 0
			drawFfpsCircle(img, make_pair(v.first, v4p2d_homo));
			drawFfpsText(img, make_pair(v.first, v4p2d_homo));
		}

		//shapemodels::MorphableModel mm = shapemodels::MorphableModel::loadScmModel("C:\\Users\\Patrik\\Documents\\GitHub\\bsl_model_first\\SurreyLowResGuosheng\\NON3448\\ShpVtxModelBin_NON3448.scm", "C:\\Users\\Patrik\\Documents\\GitHub\\featurePoints_SurreyScm.txt");
		std::shared_ptr<render::Mesh> meshToDraw = std::make_shared<render::Mesh>(mm.getMean());

		const float aspect = (float)img.cols/(float)img.rows; // 640/480
		render::Camera camera(Vec3f(0.0f, 0.0f, 0.0f), /*horizontalAngle*/0.0f*(CV_PI/180.0f), /*verticalAngle*/0.0f*(CV_PI/180.0f), render::Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, /*zNear*/-0.1f, /*zFar*/-100.0f));
		render::RenderDevicePnP r(img.cols, img.rows, camera); // 640, 480
		//r.setModelTransform(render::utils::MatrixUtils::createScalingMatrix(1.0f/140.0f, 1.0f/140.0f, 1.0f/140.0f));
		r.setIntrinsicCameraTransform(intrinsicCameraMatrix);
		r.setExtrinsicCameraTransform(extrinsicCameraMatrix);
		r.draw(meshToDraw, nullptr);
		Mat buff = r.getImage();
		Mat buffWithoutAlpha;
		//buff.convertTo(buffWithoutAlpha, CV_BGRA2BGR);
		cvtColor(buff, buffWithoutAlpha, cv::COLOR_BGRA2BGR);
		Mat weighted = img.clone(); // get the right size
		cv::addWeighted(img, 0.4, buffWithoutAlpha, 0.6, 0.0, weighted);

		vector<Point3f> faceSizePoints;
		faceSizePoints.push_back(mm.getShapeModel().getMeanAtPoint("center.nose.tip"));
		faceSizePoints.push_back(mm.getShapeModel().getMeanAtPoint("right.eye.pupil.center"));
		faceSizePoints.push_back(mm.getShapeModel().getMeanAtPoint("left.eye.pupil.center"));
		Mat faceSizePoints2D;
		cv::projectPoints(faceSizePoints, rvec, tvec, cameraMatrix, Mat(), faceSizePoints2D);
		
		Point2f nt(faceSizePoints2D.rowRange(0, 1).reshape(1));
		Point2f reyec(faceSizePoints2D.rowRange(1, 2).reshape(1));
		Point2f leyec(faceSizePoints2D.rowRange(2, 3).reshape(1));
		float ied = norm(Mat(reyec), Mat(leyec), cv::NORM_L2);
		float width = ied*1.8f;
		return Rect(nt.x - width/2.0f, nt.y - width/2.0f, width, width);
	};


private:
	MorphableModel mm;

	Point3f originVertex;

	Mat cameraMatrix;
	Mat rodrRotVec, rodrTransVec;

	map<string, Point3f> movePoints(map<string, Point3f> points, Point3f offset)
	{
		map<string, Point3f> movedPoints;
		for (const auto& p : points) {
			Point3f tmp(p.second.x + offset.x, p.second.y + offset.y, p.second.z + offset.z);
			movedPoints.insert(std::make_pair(p.first, tmp));
		}

		return movedPoints;
	}

	// POSIT expects the image center to be in the middle of the image (probably
	// because that's where the camera points to), thus we have to move all the points
	Point2f positToImgCoords(Point2f positCoords, int imgWidth, int imgHeight) {
		float centerX = ((float)(imgWidth-1))/2.0f;
		float centerY = ((float)(imgHeight-1))/2.0f;
		return Point2f(centerX + positCoords.x, centerY - positCoords.y);
	};
	Point2f imgToPositCoords(Point2f imgCoords, int imgWidth, int imgHeight) {
		float centerX = ((float)(imgWidth-1))/2.0f;
		float centerY = ((float)(imgHeight-1))/2.0f;
		return Point2f(imgCoords.x - centerX, centerY - imgCoords.y);
	};

	// Caution: The vertex should already be moved to correspond to the right origin
	Point2f positProject(Point3f vertex, Mat translation_vector, Mat rotation_matrix, float focalLength) {
		Mat vertex3d(vertex);
		Mat vertex3dproj = rotation_matrix * vertex3d;
		vertex3dproj = vertex3dproj + translation_vector.t();
		cv::Vec3f vertex3dprojvec(vertex3dproj);
		Point2f projpoint(0.0f, 0.0f);
		if (vertex3dprojvec[2] != 0) {
			projpoint.x = focalLength * vertex3dprojvec[0] / vertex3dprojvec[2];
			projpoint.y = focalLength * vertex3dprojvec[1] / vertex3dprojvec[2];
		}
		return projpoint;
	};

	map<string, Point3f> get3dmmLmsFromFfps(map<string, Point2f> ffps, MorphableModel mm)
	{
		map<string, Point3f> mmPoints;
		for (const auto& p : ffps) {
			mmPoints.insert(std::make_pair(p.first, mm.getShapeModel().getMeanAtPoint(p.first)));
		}
		return mmPoints;
	}

	void drawFfpsCircle(Mat image, pair<string, Point2f> landmarks)
	{
		cv::Point center(cvRound(landmarks.second.x), cvRound(landmarks.second.y));
		int radius = cvRound(3);
		circle(image, center, 1, cv::Scalar(0,255,0), 1, 8, 0 );	// draw the circle center
		circle(image, center, radius, cv::Scalar(0,0,255), 1, 8, 0 );	// draw the circle outline

	}

	void drawFfpsText(Mat image, pair<string, Point2f> landmarks)
	{
		cv::Point center(cvRound(landmarks.second.x), cvRound(landmarks.second.y));
		std::ostringstream text;
		int fontFace = cv::FONT_HERSHEY_PLAIN;
		double fontScale = 0.7;
		int thickness = 1;  
		text << landmarks.first << std::ends;
		putText(image, text.str(), center, fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
		text.str("");
	}

};

} /* namespace shapemodels */
#endif // FEATUREPOINTSEVALUATOR_HPP_
