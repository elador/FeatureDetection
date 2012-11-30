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

	virtual ~FeatureVector() {}

	/**
	 * @return The number of features.
	 */
	virtual unsigned int getSize() const = 0;

	/**
	 * @return The values of the features.
	 */
	virtual const float* getValues() const = 0;

	/**
	 * Retrives a certain feature.
	 *
	 * @param[in] index The index of the feature.
	 * @return The feature at the given index.
	 */
	virtual float get(int index) const = 0;

	/**
	 * Retrives a certain feature.
	 *
	 * @param[in] index The index of the feature.
	 * @return The feature at the given index.
	 */
	inline const float operator[](int index) const {
		return get(index);
	}
};

} /* namespace tracking */
#endif /* FEATUREVECTOR_H_ */
