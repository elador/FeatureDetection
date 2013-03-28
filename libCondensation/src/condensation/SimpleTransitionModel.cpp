/*
 * SimpleTransitionModel.cpp
 *
 *  Created on: 10.07.2012
 *      Author: poschmann
 */

#include "condensation/SimpleTransitionModel.hpp"
#include "condensation/Sample.hpp"
#include <ctime>
#include <cmath>

namespace condensation {

SimpleTransitionModel::SimpleTransitionModel(double scatter) : scatter(scatter),
		generator(boost::mt19937(time(0)), boost::normal_distribution<>()) {}

SimpleTransitionModel::~SimpleTransitionModel() {}

void SimpleTransitionModel::predict(Sample& sample, const vector<double>& offset) {
	double scatter = this->scatter;
	// drift using offset
	double x = sample.getX() + offset[0];
	double y = sample.getY() + offset[1];
	double s = sample.getSize() + offset[2];
	// diffuse
	double deviation = scatter * sample.getSize();
	x += deviation * generator();
	y += deviation * generator();
	s *= pow(2, scatter * generator());
	// round to integer
	sample.setX((int)(x + 0.5));
	sample.setY((int)(y + 0.5));
	sample.setSize((int)(s + 0.5));
}

} /* namespace condensation */
