/*
 * FeaturePointsRANSAC.h
 *
 *  Created on: 22.10.2012
 *      Author: Patrik Huber
 */
#pragma once

class FeaturePointsRANSAC
{
public:
	FeaturePointsRANSAC(void);
	~FeaturePointsRANSAC(void);

	void init();			// Load the 3DMM
	void runRANSAC(void* landmarkData, int minPointsToFitModel, float thresholdForDatapointFitsModel, int numClosePointsRequiredForGoodFit, int numIter);		// Run the main algorithm
	// Output:	- model - model parameters which best fit the data
	//			- consensus_set - data points from which this model has been estimated
	//			- error - the error of this model relative to the data

private:
	void* model;	// the 3DMM loaded into memory
	void fitModelToPoints(void* landmarks);	// project the points into the model (=the same as fitting the model to the points?)
								// return: The fitting/projection error

};

