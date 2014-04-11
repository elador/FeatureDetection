/*
 * StateValidator.hpp
 *
 *  Created on: 10.04.2014
 *      Author: poschmann
 */

#ifndef STATEVALIDATOR_HPP_
#define STATEVALIDATOR_HPP_

#include <vector>
#include <memory>

namespace imageprocessing {
class VersionedImage;
}

namespace condensation {

class Sample;

/**
 * Validator of the target state.
 */
class StateValidator {
public:

	virtual ~StateValidator() {}

	/**
	 * Validates the given target state.
	 *
	 * @param[in] target Target state.
	 * @param[in] samples Samples that were used for extracting the target state.
	 * @param[in] image Current image.
	 * @return True if the given state is valid, false otherwise.
	 */
	virtual bool isValid(const Sample& target, const std::vector<std::shared_ptr<Sample>>& samples,
			std::shared_ptr<imageprocessing::VersionedImage> image) = 0;
};

} /* namespace condensation */
#endif /* STATEVALIDATOR_HPP_ */
