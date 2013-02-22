/*
 * RansacFeaturePointsSelector.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef RANSACFEATUREPOINTSSELECTOR_HPP_
#define RANSACFEATUREPOINTSSELECTOR_HPP_

#include "shapemodels/FeaturePointsSelector.hpp"

namespace shapemodels {


/**
 * A ... .
 */
class RansacFeaturePointsSelector : public FeaturePointsSelector {
public:

	explicit RansacFeaturePointsSelector();

	virtual ~RansacFeaturePointsSelector();

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
#endif // RANSACFEATUREPOINTSSELECTOR_HPP_
