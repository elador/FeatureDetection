/*
 * FeaturePointsRANSAC.hpp
 *
 *  Created on: 22.10.2012
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FEATUREPOINTSMODELRANSAC_HPP_
#define FEATUREPOINTSMODELRANSAC_HPP_

#include "shapemodels/MorphableModel.hpp"

#include "imageprocessing/Patch.hpp"

#include "opencv2/core/core.hpp"

#include <vector>
#include <map>
#include <string>
#include <memory>

using cv::Mat;
using cv::Point2f;
using cv::Point3f;
using std::vector;
using std::pair;
using std::string;
using std::shared_ptr;

namespace shapemodels {

/**
 * TODO
 */
class FeaturePointsRANSAC {

public:
	
	void runRANSAC(Mat image, vector<pair<string, vector<shared_ptr<imageprocessing::Patch>> > > landmarkData, MorphableModel mm, float thresholdForDatapointFitsModel, int numIter=0, int numClosePointsRequiredForGoodFit=4, int minPointsToFitModel=3);		// Run the main algorithm
	// Output:	- model - model parameters which best fit the data
	//			- consensus_set - data points from which this model has been estimated
	//			- error - the error of this model relative to the data

	void renderModel(Mat img);


private:



	//void fitModelToPoints(void* landmarks);	// project the points into the model (=the same as fitting the model to the points?)
	// return: The fitting/projection error

	Point2f calculateCenterpoint(vector<pair<string, Point2f> >);
	Point3f calculateCenterpoint(vector<pair<string, Point3f> >);

	vector<pair<string, Point2f> > movePointsToCenter(vector<pair<string, Point2f> >, Point2f);
	vector<pair<string, Point3f> > movePointsToCenter(vector<pair<string, Point3f> >, Point3f);

	void printPointsConsole(string, vector<pair<string, Point2f> >);
	void printPointsConsole(string, vector<pair<string, Point3f> >);

	vector<pair<string, Point2f> > projectVerticesToImage(vector<string>, Point3f, Point2f, Mat, Mat, MorphableModel mm);

	void drawAndPrintLandmarksInImage(Mat, vector<pair<string, Point2f> >);
	void drawLandmarksText(Mat, vector<pair<string, Point2f> >);
	void drawFull3DMMProjection(Mat, Point3f centerIn3dmm, Point2f centerInImage, Mat s, Mat t, MorphableModel mm);

	vector<pair<string, Point3f> > get3dmmLmsFromFfps(vector<pair<string, Point2f>>, MorphableModel mm);

	pair<Mat, Mat> calculateProjection(vector<pair<string, Point2f> >, vector<pair<string, Point3f> >);

	struct LandmarksVerticesPair {
		vector<pair<string, Point2f> > landmarks;
		Point2f landmarksMean;		// Mw
		vector<pair<string, Point3f> > vertices;
		Point3f verticesMean;		// Mv
	};

	LandmarksVerticesPair getCorrespondingLandmarkpointsAtOriginFrom3DMM(vector<pair<string, Point2f>>, MorphableModel mm);
};

} /* namespace shapemodels */
#endif /* FEATUREPOINTSMODELRANSAC_HPP_ */
