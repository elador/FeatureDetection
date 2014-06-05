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

using cv::Mat;
using std::vector;
using std::shared_ptr;

namespace condensation {

SimpleTransitionModel::SimpleTransitionModel(double positionDeviation, double sizeDeviation) :
		positionDeviation(positionDeviation), sizeDeviation(sizeDeviation),
		generator(boost::mt19937(time(0)), boost::normal_distribution<>()) {}

void SimpleTransitionModel::init(const Mat& image) {}

void SimpleTransitionModel::predict(vector<shared_ptr<Sample>>& samples, const Mat& image, const shared_ptr<Sample> target) {
	for (shared_ptr<Sample> sample : samples) {
		// add noise to velocity
		double vx = sample->getVx();
		double vy = sample->getVy();
		double vs = sample->getVSize();
		// diffuse
		vx += positionDeviation * generator();
		vy += positionDeviation * generator();
		vs *= pow(2, sizeDeviation * generator());
		// round to integer
		sample->setVx(static_cast<int>(std::round(vx)));
		sample->setVy(static_cast<int>(std::round(vy)));
		sample->setVSize(vs);
		// change position according to velocity
		sample->setX(sample->getX() + sample->getVx());
		sample->setY(sample->getY() + sample->getVy());
		sample->setSize(static_cast<int>(std::round(sample->getSize() * sample->getVSize())));
	}
}

} /* namespace condensation */
