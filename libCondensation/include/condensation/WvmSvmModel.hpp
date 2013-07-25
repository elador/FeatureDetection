/*
 * WvmSvmModel.hpp
 *
 *  Created on: 25.07.2012
 *      Author: poschmann
 */

#ifndef WVMSVMMODEL_HPP_
#define WVMSVMMODEL_HPP_

#include "condensation/MeasurementModel.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

using std::shared_ptr;
using std::unordered_map;
using std::pair;

namespace imageprocessing {
class FeatureExtractor;
class Patch;
}
using imageprocessing::FeatureExtractor;
using imageprocessing::Patch;

namespace classification {
class ProbabilisticWvmClassifier;
class ProbabilisticSvmClassifier;
}
using classification::ProbabilisticWvmClassifier;
using classification::ProbabilisticSvmClassifier;

namespace condensation {

/**
 * Measurement model that uses a WVM for quick elimination and evaluates the samples that remain after an
 * overlap elimination with a SVM. The weight of the samples will be the product of the certainties from
 * the two detectors, they will be regarded as being independent (although they are not). The certainties
 * for the SVM of samples that are not evaluated by it will be chosen to be 0.5 (unknown).
 */
class WvmSvmModel : public MeasurementModel {
public:

	/**
	 * Constructs a new WVM-SVM measurement model.
	 *
	 * @param[in] featureExtractor The feature extractor.
	 * @param[in] wvm The fast WVM.
	 * @param[in] svm The slower SVM.
	 * TODO overlap elimination?
	 */
	WvmSvmModel(shared_ptr<FeatureExtractor> featureExtractor,
			shared_ptr<ProbabilisticWvmClassifier> wvm, shared_ptr<ProbabilisticSvmClassifier> svm);

	~WvmSvmModel();

	void update(shared_ptr<VersionedImage> image);

	void evaluate(Sample& sample);

	void evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples);

private:

	shared_ptr<FeatureExtractor> featureExtractor; ///< The feature extractor.
	shared_ptr<ProbabilisticWvmClassifier> wvm;    ///< The fast WVM.
	shared_ptr<ProbabilisticSvmClassifier> svm;    ///< The slower SVM.
	//shared_ptr<OverlapElimination> oe;      ///< The overlap elimination algorithm. TODO
	unordered_map<shared_ptr<Patch>, pair<bool, double>> cache; ///< The cache of the WVM classification results.
};

} /* namespace condensation */
#endif /* WVMSVMMODEL_HPP_ */
