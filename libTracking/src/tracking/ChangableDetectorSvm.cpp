/*
 * ChangableDetectorSvm.cpp
 *
 *  Created on: 01.08.2012
 *      Author: poschmann
 */

#include "tracking/ChangableDetectorSvm.h"

namespace tracking {

ChangableDetectorSvm::ChangableDetectorSvm() {}

ChangableDetectorSvm::~ChangableDetectorSvm() {}

void ChangableDetectorSvm::changeRbfParameters(int numSv, unsigned char** supportVectors, float* alphas, float rho,
		float gamma, float probParamA, float probParamB) {
	for (int i = 0; i < this->numSV; ++i)
		delete[] this->support[i];
	delete[] this->support;
	delete[] this->alpha;

	this->numSV = numSv;
	this->support = supportVectors;
	this->alpha = alphas;
	this->nonlin_threshold = rho;
	this->nonLinType = 2;
	this->basisParam = gamma;
	this->posterior_svm[0] = probParamA;
	this->posterior_svm[1] = probParamB;
}

} /* namespace tracking */
