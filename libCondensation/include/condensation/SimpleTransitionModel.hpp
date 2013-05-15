/*
 * SimpleTransitionModel.hpp
 *
 *  Created on: 10.07.2012
 *      Author: poschmann
 */

#ifndef SIMPLETRANSITIONMODEL_HPP_
#define SIMPLETRANSITIONMODEL_HPP_

#include "condensation/TransitionModel.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/variate_generator.hpp"

namespace condensation {

/**
 * Simple transition model that diffuses position and velocity.
 */
class SimpleTransitionModel : public TransitionModel {
public:

	/**
	 * Constructs a new simple transition model.
	 *
	 * @param[in] positionScatter The scatter that controls the diffusion of the position.
	 * @param[in] velocityScatter The scatter that controls the diffusion of the velocity.
	 */
	explicit SimpleTransitionModel(double positionScatter = 0.05, double velocityScatter = 0.1);

	~SimpleTransitionModel();

	void predict(Sample& sample);

	/**
	 * @return The scatter that controls the diffusion of the position.
	 */
	double getPositionScatter() {
		return positionScatter;
	}

	/**
	 * @param[in] The new scatter that controls the diffusion of the position.
	 */
	void setPositionScatter(double scatter) {
		this->positionScatter = scatter;
	}

	/**
	 * @return The scatter that controls the diffusion of the velocity.
	 */
	double getVelocityScatter() {
		return velocityScatter;
	}

	/**
	 * @param[in] The new scatter that controls the diffusion of the velocity.
	 */
	void setVelocityScatter(double scatter) {
		this->velocityScatter = scatter;
	}

private:
	double positionScatter = 0.1; ///< The scatter that controls the diffusion of the position.
	double velocityScatter = 0.1; ///< The scatter that controls the diffusion of the velocity.
	boost::variate_generator<boost::mt19937, boost::normal_distribution<> > generator; ///< Random number generator.
};

} /* namespace condensation */
#endif /* SIMPLETRANSITIONMODEL_HPP_ */
