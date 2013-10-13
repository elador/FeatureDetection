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

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/legacy/compat.hpp"

#include <memory>
#include <iostream>
#include <utility>

using cv::Point2f;
using cv::Point3f;

using std::unique_ptr;
using std::pair;

namespace shapemodels {

/**
 * A ... .
 * Note for POSIT: I think it would be better to use 3 points and a simple "rigid projection" of
 * the model (as in Sami's paper). Then we could also do a fast reject, and have a proper RANSAC
 * around it.
 * Note2: The first point is always used as camera center I think. Try using 3 points and
 * the center of those 3 in the model as a 4th (first) ?
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

	std::pair<Mat, Mat> doPosit(map<string, shared_ptr<imageprocessing::Patch>> landmarkPoints, Mat img) { // img for debug purposes
		float error = 0.0f;

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

	float evaluateNewPoint(string landmarkName, shared_ptr<imageprocessing::Patch> patch, Mat trans, Mat rot, Mat img) {
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

private:
	MorphableModel mm;

	Point3f originVertex;

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
			int theVertexToGet = mm.getShapeModel().getFeaturePointsMap()[p.first];
			Point3f tmp(mm.getShapeModel().getMean()[3*theVertexToGet+0], mm.getShapeModel().getMean()[3*theVertexToGet+1], mm.getShapeModel().getMean()[3*theVertexToGet+2]); // TODO! Does this start to count at 0 or 1 ? At 0. (99.999% sure)
			mmPoints.insert(std::make_pair(p.first, tmp));
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
