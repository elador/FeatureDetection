/*
 * FeatureVector.cpp
 *
 *  Created on: 22.11.2012
 *      Author: poschmann
 */

#include "classification/FeatureVector.h"

namespace classification {

FeatureVector::FeatureVector(int size) : size(size), values(new float[size]) {}

FeatureVector::FeatureVector(const FeatureVector& other) : size(other.size), values(new float[size]) {
	for (int i = 0; i < size; ++i)
		values[i] = other.values[i];
}

FeatureVector::~FeatureVector() {
	delete[] values;
}

FeatureVector& FeatureVector::operator=(const FeatureVector& other) {
	float* originalValues = values;
	values = new float[other.size];
	for (int i = 0; i < other.size; ++i)
		values[i] = other.values[i];
	size = other.size;
	delete[] originalValues;
	return *this;
}

} /* namespace classification */
