/*
 * EagerFeatureVector.cpp
 *
 *  Created on: 22.11.2012
 *      Author: poschmann
 */

#include "classification/EagerFeatureVector.h"

namespace classification {

EagerFeatureVector::EagerFeatureVector(int size) : size(size), values(new float[size]) {}

EagerFeatureVector::EagerFeatureVector(const EagerFeatureVector& other) : size(other.size), values(new float[size]) {
	for (int i = 0; i < size; ++i)
		values[i] = other.values[i];
}

EagerFeatureVector::~EagerFeatureVector() {
	delete[] values;
}

EagerFeatureVector& EagerFeatureVector::operator=(const EagerFeatureVector& other) {
	float* originalValues = values;
	values = new float[other.size];
	for (int i = 0; i < other.size; ++i)
		values[i] = other.values[i];
	size = other.size;
	delete[] originalValues;
	return *this;
}

} /* namespace tracking */
