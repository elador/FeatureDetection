/*
 * TrainableTwoStageClassifier.cpp
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#include "classification/TrainableTwoStageClassifier.h"

namespace classification {

TrainableTwoStageClassifier::TrainableTwoStageClassifier(shared_ptr<Classifier> first,
		shared_ptr<TrainableClassifier> second) : TwoStageClassifier(first, second), trainable(second) {}

TrainableTwoStageClassifier::~TrainableTwoStageClassifier() {}

} /* namespace classification */
