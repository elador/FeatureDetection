/*
 * FeaturePointsRANSAC.h
 *
 *  Created on: 22.10.2012
 *      Author: Patrik Huber
 */
#pragma once

#include "FdPatch.h"

#include <opencv2/core/core.hpp>

#include <vector>
#include <map>

class FdImage;

class FeaturePointsRANSAC
{
public:
	FeaturePointsRANSAC(void);
	~FeaturePointsRANSAC(void);

	void init(void);			// Load the 3DMM & featurePoints file
	void runRANSAC(FdImage*, std::vector<std::pair<std::string, std::vector<FdPatch*> > > landmarkData, float thresholdForDatapointFitsModel, int numIter=0, int numClosePointsRequiredForGoodFit=4, int minPointsToFitModel=3);		// Run the main algorithm
	// Output:	- model - model parameters which best fit the data
	//			- consensus_set - data points from which this model has been estimated
	//			- error - the error of this model relative to the data

	void renderModel(FdImage* img);

private:
	std::vector<float> modelMeanShp;	// the 3DMM mean shape loaded into memory. Data is XYZXYZXYZ...
	std::vector<float> modelMeanTex;
	std::map<std::string, int> featurePointsMap;	// Holds the translation from feature point name (e.g. reye) to the vertex number in the model

	void loadModel(std::string);
	void loadFeaturePoints(std::string);


	void fitModelToPoints(void* landmarks);	// project the points into the model (=the same as fitting the model to the points?)
								// return: The fitting/projection error

	cv::Point2f calculateCenterpoint(std::vector<std::pair<std::string, cv::Point2f> >);
	cv::Point3f calculateCenterpoint(std::vector<std::pair<std::string, cv::Point3f> >);

	std::vector<std::pair<std::string, cv::Point2f> > movePointsToCenter(std::vector<std::pair<std::string, cv::Point2f> >, cv::Point2f);
	std::vector<std::pair<std::string, cv::Point3f> > movePointsToCenter(std::vector<std::pair<std::string, cv::Point3f> >, cv::Point3f);

	void printPointsConsole(std::string, std::vector<std::pair<std::string, cv::Point2f> >);
	void printPointsConsole(std::string, std::vector<std::pair<std::string, cv::Point3f> >);

	std::vector<std::pair<std::string, cv::Point2f> > projectVerticesToImage(std::vector<std::string>, cv::Point3f, cv::Point2f, cv::Mat, cv::Mat);

	void drawAndPrintLandmarksInImage(cv::Mat, std::vector<std::pair<std::string, cv::Point2f> >);
	void drawLandmarksText(cv::Mat, std::vector<std::pair<std::string, cv::Point2f> >);
	void drawFull3DMMProjection(cv::Mat, cv::Point3f centerIn3dmm, cv::Point2f centerInImage, cv::Mat s, cv::Mat t);

	std::vector<std::pair<std::string, cv::Point3f> > get3dmmLmsFromFfps(std::vector<std::pair<std::string, cv::Point2f> >);

	std::pair<cv::Mat, cv::Mat> calculateProjection(std::vector<std::pair<std::string, cv::Point2f> >, std::vector<std::pair<std::string, cv::Point3f> >);

	struct LandmarksVerticesPair {
		std::vector<std::pair<std::string, cv::Point2f> > landmarks;
		cv::Point2f landmarksMean;		// Mw
		std::vector<std::pair<std::string, cv::Point3f> > vertices;
		cv::Point3f verticesMean;		// Mv
	};

	LandmarksVerticesPair getCorrespondingLandmarkpointsAtOriginFrom3DMM(std::vector<std::pair<std::string, cv::Point2f> >);

};
