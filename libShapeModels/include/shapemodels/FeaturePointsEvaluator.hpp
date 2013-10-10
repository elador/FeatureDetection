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

using cv::Point2f;
using cv::Point3f;
using std::unique_ptr;

namespace shapemodels {

/**
 * A ... .
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

	std::pair<Mat, Mat> evaluate(map<string, shared_ptr<imageprocessing::Patch>> landmarkPoints, Mat img) { // img for debug purposes
		float error = 0.0f;

		// Convert patch to Point2f
		map<string, Point2f> landmarkPointsAsPoint2f;
		for (const auto& p : landmarkPoints) {
			landmarkPointsAsPoint2f.insert(make_pair(p.first, Point2f(p.second->getX(), p.second->getY())));
		}

		map<string, Point3f> mmVertices = get3dmmLmsFromFfps(landmarkPointsAsPoint2f, mm);

		mmVertices = movePointsToOrigin(mmVertices, begin(mmVertices)->second);

		//Create the model points
		vector<CvPoint3D32f> modelPoints;
		/*modelPoints.push_back(cvPoint3D32f(0.0f, 0.0f, 0.0f)); //The first must be (0,0,0)
		modelPoints.push_back(cvPoint3D32f(0.0f, 0.0f, 1.0f));
		modelPoints.push_back(cvPoint3D32f(1.0f, 0.0f, 0.0f));
		modelPoints.push_back(cvPoint3D32f(0.0f, 1.0f, 0.0f));*/
		for (const auto& v : mmVertices) {
			modelPoints.push_back(cvPoint3D32f(v.second.x, v.second.y, v.second.z));
		}

		//Create the image points 
		vector<CvPoint2D32f> srcImagePoints;
		for (const auto& p : landmarkPoints) {
			srcImagePoints.push_back(cvPoint2D32f(p.second->getX(), p.second->getY()));
		}

		//Create the POSIT object with the model points
		//unique_ptr<CvPOSITObject, decltype(cvReleasePOSITObject)> positObject;
		//positObject = unique_ptr<CvPOSITObject, decltype(cvReleasePOSITObject)>(cvCreatePOSITObject(&modelPoints[0], (int)modelPoints.size()), cvReleasePOSITObject);

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
		}
		for (const auto& v : mmVertices) {
			cv::Mat vertex3d(v.second);
			cv::Mat vertex3dproj = rotation_matrix * vertex3d;
			vertex3dproj = vertex3dproj + translation_vector;
			cv::Point2f projpoint();
			//projpoint.x
			//cv::rectangle(img, cv::Point(cvRound(p.second->getX()-2.0f), cvRound(p.second->getY()-2.0f)), cv::Point(cvRound(p.second->getX()+2.0f), cvRound(p.second->getY()+2.0f)), cv::Scalar(255, 0, 0));
		}
		
			
		return std::make_pair(translation_vector, rotation_matrix);
	};

private:
	MorphableModel mm;

	map<string, Point3f> movePointsToOrigin(map<string, Point3f> points, Point3f origin)
	{
		map<string, Point3f> movedPoints;
		for (const auto& p : points) {
			Point3f tmp(p.second.x - origin.x, p.second.y - origin.y, p.second.z - origin.z);
			movedPoints.insert(std::make_pair(p.first, tmp));
		}

		return movedPoints;
	}

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

};

} /* namespace shapemodels */
#endif // FEATUREPOINTSEVALUATOR_HPP_
