/*
 * FeaturePointsModel.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef FEATUREPOINTSMODEL_HPP_
#define FEATUREPOINTSMODEL_HPP_

#include "shapemodels/FeaturePointsSelector.hpp"
#include "shapemodels/FeaturePointsEvaluator.hpp"

#include "imageprocessing/Patch.hpp"

#include <memory>
#include <map>
#include <string>
#include <vector>

using std::unique_ptr;
using std::shared_ptr;
using std::string;
using std::vector;
using std::map;

namespace shapemodels {

/**
 * A model that is able to, given a list of several feature point candidates, select the most likely point combinations by capturing the relationship between different feature points.
 * We could make this templated on the patch-type to operate on, e.g. map<string, T> and T is shared_ptr<imageprocessing::Patch>, Vec2f, etc.
 * Or maybe here it makes more sense to use function overloading. For the Evaluator maybe templates would make more sense.
 */
class FeaturePointsModel {

public:
	FeaturePointsModel() {};


	FeaturePointsModel(FeaturePointsSelector selector, FeaturePointsEvaluator evaluator) : selector(selector), evaluator(evaluator) {

	};


	/**
	 * A getter.
	 *
	 * @return Something.
	 */
	//virtual const int getter() const;

	/**
	 * Does xyz
	 *
	 * @param[in] param The parameter.
	 */
	// Problem: This needs arguments that are dependent on the child class. So we should encapsulate this as Algorithm or something?
	virtual map<string, shared_ptr<imageprocessing::Patch>> run(Mat img, float thresholdForDatapointFitsModel/*=30.0f*/, int numIter/*=0*/, int numClosePointsRequiredForGoodFit/*=4*/, int minPointsToFitModel/*=3*/) = 0; // Todo: Make the args optional
		
		//The whole algorithm here. evaluator only evaluates 1 kombo. All logic here. Selector provides functions to get random items (1 or several) etc.
		
		//map<string, vector<shared_ptr<imageprocessing::Patch>>> featureSelection;
		//do {
			//featureSelection.clear(); // Not necessary?
		//	featureSelection = selector->select();
		//} while (evaluator->evaluate(featureSelection));
		
	void setLandmarks(map<string, vector<shared_ptr<imageprocessing::Patch>>> landmarks) { // Note: That's maybe not so nice because this abstract class is now fixed on this type. But then again, what other type than these landmarks would we use. All here is based on that. So maybe typedef that.
		selector.setLandmarks(landmarks);
	};

protected: // That's supposedly not good style since we should keep total control over our own class variables
	FeaturePointsSelector selector;
	FeaturePointsEvaluator evaluator;

};

} /* namespace shapemodels */
#endif // FEATUREPOINTSMODEL_HPP_
