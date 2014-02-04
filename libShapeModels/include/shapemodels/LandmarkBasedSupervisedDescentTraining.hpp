/*
 * LandmarkBasedSupervisedDescentTraining.hpp
 *
 *  Created on: 04.02.2014
 *      Author: Patrik Huber
 */

#pragma once

#ifndef LANDMARKBASEDSUPERVISEDDESCENTTRAINING_HPP_
#define LANDMARKBASEDSUPERVISEDDESCENTTRAINING_HPP_

#include "shapemodels/PcaModel.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/property_tree/ptree.hpp"

namespace shapemodels {

/**
 * Desc
 */
class LandmarkBasedSupervisedDescentTraining  {
public:

	/**
	 * Constructs a new Morphable Model.
	 *
	 * @param[in] a b
	 */
	LandmarkBasedSupervisedDescentTraining() {};


private:


};

} /* namespace shapemodels */
#endif /* LANDMARKBASEDSUPERVISEDDESCENTTRAINING_HPP_ */
