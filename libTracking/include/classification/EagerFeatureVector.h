/*
 * EagerFeatureVector.h
 *
 *  Created on: 22.11.2012
 *      Author: poschmann
 */

#ifndef EAGERFEATUREVECTOR_H_
#define EAGERFEATUREVECTOR_H_

#include "classification/FeatureVector.h"
#include <vector>

using std::vector;

namespace classification {

/**
 * Feature vector that stores the computed values.
 */
class EagerFeatureVector : public FeatureVector {
public:

	/**
	 * Constructs a new eager feature vector with the given size and default values.
	 *
	 * @param[in] size The number of features.
	 */
	explicit EagerFeatureVector(int size);

	/**
	 * Constructs a copy of another eager feature vector.
	 *
	 * @param[in] other The other eager feature vector.
	 */
	EagerFeatureVector(const EagerFeatureVector& other);

	~EagerFeatureVector();

	/**
	 * Assignment operator.
	 *
	 * @param[in] other The eager feature vector to copy all values from.
	 */
	EagerFeatureVector& operator=(const EagerFeatureVector& other);

	inline unsigned int getSize() const {
		return size;
	}

	inline const float* getValues() const {
		return values;
	}

	inline float get(int index) const {
		return values[index];
	}

	/**
	 * Changes the value of a certain feature.
	 *
	 * @param[in] index The index of the feature.
	 * @param[in] value The new feature value.
	 */
	inline void set(int index, float value) {
		values[index] = value;
	}

	/**
	 * Retrives the reference of a certain feature value.
	 *
	 * @param[in] index The index of the feature.
	 * @return The reference of the feature value at the given index.
	 */
	inline float& at(int index) {
		return values[index];
	}

	/**
	 * Retrives the reference of a certain feature value.
	 *
	 * @param[in] index The index of the feature.
	 * @return The reference of the feature value at the given index.
	 */
	inline float& operator[](int index) {
		return at(index);
	}

private:

	int size;      ///< The number of features.
	float* values; ///< The values of the features.
};

} /* namespace tracking */
#endif /* EAGERFEATUREVECTOR_H_ */
