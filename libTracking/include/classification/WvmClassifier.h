/*
 * WvmClassifier.h
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#ifndef WVMCLASSIFIER_H_
#define WVMCLASSIFIER_H_

#include "classification/Classifier.h"
#include "boost/shared_ptr.hpp"

using boost::shared_ptr;

class VDetectorVectorMachine;

namespace classification {

/**
 * Classifier based on the Wavelet Reduced Vector Machine of libFeatureDetection.
 */
class WvmClassifier : public Classifier {
public:

	/**
	 * Constructs a new WVM classifier.
	 *
	 * @param[in] wvm The WVM.
	 */
	explicit WvmClassifier(shared_ptr<VDetectorVectorMachine> wvm);

	~WvmClassifier();

	pair<bool, double> classify(const FeatureVector& featureVector) const;

private:

	shared_ptr<VDetectorVectorMachine> wvm; ///< The WVM.
};

} /* namespace classification */
#endif /* WVMCLASSIFIER_H_ */
