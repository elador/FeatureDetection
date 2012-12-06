/*
 * FeatureVector.h
 *
 *  Created on: 22.11.2012
 *      Author: poschmann
 */

#ifndef FEATUREVECTOR_H_
#define FEATUREVECTOR_H_

namespace classification {

/**
 * Feature vector containing float values.
 */
class FeatureVector {
public:

	/**
	 * Constructs a new feature vector with the given size and default values.
	 *
	 * @param[in] size The number of features.
	 */
	explicit FeatureVector(int size);

	/**
	 * Constructs a copy of another feature vector.
	 *
	 * @param[in] other The other feature vector.
	 */
	FeatureVector(const FeatureVector& other);

	~FeatureVector();

	/**
	 * Assignment operator.
	 *
	 * @param[in] other The feature vector to copy all values from.
	 */
	FeatureVector& operator=(const FeatureVector& other);

	/**
	 * @return The number of features.
	 */
	inline unsigned int getSize() const {
		return size;
	}

	/**
	 * Retrives a certain feature.
	 *
	 * @param[in] index The index of the feature.
	 * @return The feature at the given index.
	 */
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
	 * Retrives a certain feature.
	 *
	 * @param[in] index The index of the feature.
	 * @return The feature at the given index.
	 */
	inline const float operator[](int index) const {
		return get(index);
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
#endif /* FEATUREVECTOR_H_ */
