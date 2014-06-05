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

namespace imageprocessing {
class FeatureExtractor;
class Patch;
}

namespace classification {
class ProbabilisticWvmClassifier;
class ProbabilisticSvmClassifier;
}

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
	WvmSvmModel(std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor,
			std::shared_ptr<classification::ProbabilisticWvmClassifier> wvm, std::shared_ptr<classification::ProbabilisticSvmClassifier> svm);

	void update(std::shared_ptr<imageprocessing::VersionedImage> image);

	void evaluate(Sample& sample) const;

	void evaluate(std::shared_ptr<imageprocessing::VersionedImage> image, std::vector<std::shared_ptr<Sample>>& samples);

private:

	std::shared_ptr<imageprocessing::FeatureExtractor> featureExtractor; ///< The feature extractor.
	std::shared_ptr<classification::ProbabilisticWvmClassifier> wvm; ///< The fast WVM.
	std::shared_ptr<classification::ProbabilisticSvmClassifier> svm; ///< The slower SVM.
	//std::shared_ptr<imageprocessing::OverlapElimination> oe; ///< The overlap elimination algorithm. TODO
	mutable std::unordered_map<std::shared_ptr<imageprocessing::Patch>, std::pair<bool, double>> cache; ///< The cache of the WVM classification results.
};

} /* namespace condensation */
#endif /* WVMSVMMODEL_HPP_ */
