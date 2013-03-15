/*
 * SimpleTransitionModel.h
 *
 *  Created on: 10.07.2012
 *      Author: poschmann
 */

#ifndef SIMPLETRANSITIONMODEL_H_
#define SIMPLETRANSITIONMODEL_H_

#include "condensation/TransitionModel.h"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/variate_generator.hpp"

namespace condensation {

/**
 * Transition model that linearly moves the samples and diffuses them.
 */
class SimpleTransitionModel : public TransitionModel {
public:

	/**
	 * Constructs a new simple transition model.
	 *
	 * @param[in] The scatter that controls the diffusion.
	 */
	explicit SimpleTransitionModel(double scatter = 0.25);

	~SimpleTransitionModel();

	void predict(Sample& sample, const vector<double>& offset);

	/**
	 * @return The scatter that controls the diffusion.
	 */
	inline double getScatter() {
		return scatter;
	}

	/**
	 * @param[in] The new scatter that controls the diffusion.
	 */
	inline void setScatter(double scatter) {
		this->scatter = scatter;
	}

private:
	double scatter; ///< The scatter that controls the diffusion.
	boost::variate_generator<boost::mt19937, boost::normal_distribution<> > generator; ///< Random number generator.
};

} /* namespace condensation */
#endif /* SIMPLETRANSITIONMODEL_H_ */
