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

SimpleTransitionModel::SimpleTransitionModel(double positionScatter, double velocityScatter) :
		positionScatter(positionScatter),
		velocityScatter(velocityScatter),
		generator(boost::mt19937(time(0)),
		boost::normal_distribution<>()) {}

SimpleTransitionModel::~SimpleTransitionModel() {}

void SimpleTransitionModel::predict(Sample& sample) {
	// change position according to velocity plus noise
	double x = sample.getX() + sample.getVx();
	double y = sample.getY() + sample.getVy();
	double s = sample.getSize() + sample.getVSize();
	// diffuse
	double positionDeviation = positionScatter * sample.getSize();
	x += positionDeviation * generator();
	y += positionDeviation * generator();
	s *= pow(2, positionScatter * generator());
	// round to integer
	sample.setX((int)(x + 0.5));
	sample.setY((int)(y + 0.5));
	sample.setSize((int)(s + 0.5));

	// change velocity according to noise (assumed to be constant)
	double vx = sample.getVx();
	double vy = sample.getVy();
	double vs = sample.getVSize();
	// diffuse
	double velocityDeviation = velocityScatter * sample.getSize();
	vx += velocityDeviation * generator();
	vy += velocityDeviation * generator();
	vs += velocityDeviation * generator();
	// round to integer
	sample.setVx((int)(vx + 0.5));
	sample.setVy((int)(vy + 0.5));
	sample.setVSize((int)(vs + 0.5));
}

} /* namespace condensation */
