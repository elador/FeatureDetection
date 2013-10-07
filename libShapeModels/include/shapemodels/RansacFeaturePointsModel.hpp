/*
 * RansacFeaturePointsModel.hpp
 *
 *  Created on: 03.10.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef RANSACFEATUREPOINTSMODEL_HPP_
#define RANSACFEATUREPOINTSMODEL_HPP_

#include "shapemodels/FeaturePointsModel.hpp"


namespace shapemodels {

/**
 * A model that is able to, given a list of several feature point candidates, select the most likely point combinations by capturing the relationship between different feature points.
 * Shouldn't have 'Posit' in its name as it's more generic and can use whatever Evaluator/Selector? Something not optimal here?
 * Maybe this way it makes sense:
 *   - FeaturePointsModel: RansacFeaturePointsModel
 *   - FeaturePointsSelector: Random or List (for debugging)
 *   - FeaturePointsEvaluator: POSIT (with 3DMM) or Affine?Projection (own, with 3DMM)
 */
class RansacFeaturePointsModel : public FeaturePointsModel {

public:
	RansacFeaturePointsModel(unique_ptr<FeaturePointsSelector> selector, unique_ptr<FeaturePointsEvaluator> evaluator) : FeaturePointsModel(std::move(selector), std::move(evaluator)) {

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
	map<string, shared_ptr<imageprocessing::Patch>> run(Mat img, float thresholdForDatapointFitsModel/*=30.0f*/, int numIter/*=0*/, int numClosePointsRequiredForGoodFit/*=4*/, int minPointsToFitModel/*=3*/);

private:


};

} /* namespace shapemodels */
#endif // RANSACFEATUREPOINTSMODEL_HPP_
