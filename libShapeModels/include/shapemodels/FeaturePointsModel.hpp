/*
 * FeaturePointsModel.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef FEATUREPOINTSMODEL_HPP_
#define FEATUREPOINTSMODEL_HPP_


namespace shapemodels {

/**
 * A model that is able to, given a list of several feature point candidates, select the most likely point combinations by capturing the relationship between different feature points.
 */
class FeaturePointsModel {
public:

	virtual ~FeaturePointsModel();

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


};

} /* namespace shapemodels */
#endif // FEATUREPOINTSMODEL_HPP_
