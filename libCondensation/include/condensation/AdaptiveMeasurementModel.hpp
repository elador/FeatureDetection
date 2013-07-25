/*
 * AdaptiveMeasurementModel.hpp
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#ifndef ADAPTIVEMEASUREMENTMODEL_HPP_
#define ADAPTIVEMEASUREMENTMODEL_HPP_

#include "condensation/MeasurementModel.hpp"

namespace condensation {

/**
 * Measurement model that is able to adapt itself to the target.
 */
class AdaptiveMeasurementModel : public MeasurementModel {
public:

	virtual ~AdaptiveMeasurementModel() {}

	using MeasurementModel::update;

	using MeasurementModel::evaluate;

	/**
	 * @return True if this measurement model may be used, false otherwise.
	 */
	virtual bool isUsable() = 0;

	/**
	 * Adapts this model to the given target.
	 *
	 * @param[in] image The image.
	 * @param[in] samples The weighted samples.
	 * @param[in] target The estimated target position.
	 * @return True if the model was updated, false otherwise.
	 */
	virtual bool adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples, const Sample& target) = 0;

	/**
	 * Adapts this model in case the target was not found.
	 *
	 * @param[in] image The image.
	 * @param[in] samples The weighted samples.
	 * @return True if the model was updated, false otherwise.
	 */
	virtual bool adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples) = 0;

	/**
	 * Resets this model to its original state.
	 */
	virtual void reset() = 0;
};

} /* namespace condensation */
#endif /* ADAPTIVEMEASUREMENTMODEL_HPP_ */
