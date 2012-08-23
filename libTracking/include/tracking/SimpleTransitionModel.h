/*
 * SimpleTransitionModel.h
 *
 *  Created on: 10.07.2012
 *      Author: poschmann
 */

#ifndef SIMPLETRANSITIONMODEL_H_
#define SIMPLETRANSITIONMODEL_H_

#include "tracking/TransitionModel.h"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/variate_generator.hpp"

namespace tracking {

/**
 * Transition model that linearly moves the samples and diffuses them.
 */
class SimpleTransitionModel : public tracking::TransitionModel {
public:

	/**
	 * Constructs a new simple transition model.
	 *
	 * @param[in] The scatter that controls the diffusion.
	 */
	SimpleTransitionModel(double scatter = 0.25);

	~SimpleTransitionModel();

	void predict(Sample& sample, const std::vector<double>& offset);

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

} /* namespace tracking */
#endif /* SIMPLETRANSITIONMODEL_H_ */
