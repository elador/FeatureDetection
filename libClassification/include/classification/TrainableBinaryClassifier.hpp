/*
 * TrainableBinaryClassifier.hpp
 *
 *  Created on: 05.03.2013
 *      Author: poschmann
 */

#ifndef TRAINABLEBINARYCLASSIFIER_HPP_
#define TRAINABLEBINARYCLASSIFIER_HPP_

#include "classification/TrainableClassifier.hpp"
#include "classification/BinaryClassifier.hpp"

namespace classification {

/**
 * Binary classifier that may be re-trained using new examples. Re-training is an incremental procedure
 * that adds new examples and refines the classifier. Nevertheless, it may be possible that previous
 * training examples are forgotten to ensure the classifier stays relatively small and efficient.
 */
class TrainableBinaryClassifier : public TrainableClassifier, public BinaryClassifier {
public:

	virtual ~TrainableBinaryClassifier() {}
};

} /* namespace classification */
#endif /* TRAINABLEBINARYCLASSIFIER_HPP_ */
