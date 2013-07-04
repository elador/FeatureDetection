/*
 * RvmClassifier.hpp
 *
 *  Created on: 14.06.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef RVMCLASSIFIER_HPP_
#define RVMCLASSIFIER_HPP_

#include "classification/VectorMachineClassifier.hpp"
#include "opencv2/core/core.hpp"
#include "boost/property_tree/ptree.hpp"
#include <string>
#include <vector>

using boost::property_tree::ptree;
using cv::Mat;
using std::string;
using std::vector;

namespace classification {

/**
 * Classifier based on a Reduced Vector Machine.
 * An RVM differs from an SVM in certain points:
 *  - An RVM consists of a smaller number of reduced vectors (that are usually
 *    not a subset of the training data anymore)
 *  - It has a different set of coefficients for every filter-level in the cascade.
 *    Example: RSV n has its own array of weights for all vectors 1 to n.
 *  - Each filter-level has its own threshold at which it classifies a patch.
 *    Usually, different threshold files are available that have different FAR/FRR properties.
 *  - It can be run cascaded, that is it rejects featureVector that are very unlikely to be
 *    positive very early.
 */
class RvmClassifier : public VectorMachineClassifier {
public:

	/**
	 * Constructs a new RVM classifier.
	 *
	 * @param[in] kernel The kernel function.
	 */
	explicit RvmClassifier(shared_ptr<Kernel> kernel, bool cascadedCoefficients = true);

	~RvmClassifier();

	bool classify(const Mat& featureVector) const;

	/**
	 * Determines the classification result given the distance of a feature vector to the decision hyperplane.
	 *
	 * @param[in] hyperplaneDistance The distance of a feature vector to the decision hyperplane.
	 * @param[in] filterLevel The filter-level (i.e. the reduced vector in the cascade) at which to classify.
	 * @return True if feature vectors of the given distance would be classified positively, false otherwise.
	 */
	bool classify(double hyperplaneDistance, const int filterLevel) const;

	/**
	 * Computes the distance of a feature vector to the decision hyperplane. This is the real distance without
	 * any influence by the offset for configuring the operating point of the RVM.
	 *
	 * @param[in] featureVector The feature vector.
	 * @param[in] filterLevel The filter-level (i.e. the reduced vector in the cascade) at which to classify.
	 * @return The distance of the feature vector to the decision hyperplane.
	 */
	double computeHyperplaneDistance(const Mat& featureVector, const int filterLevel) const;

	double computeHyperplaneDistanceCached(const Mat& featureVector, const int filterLevel, vector<double>& filterEvalCache) const;

	/**
	 * Returns the number of filters (RSVs) this RVM is currently using for classifying.
	 *
	 * @return The number of filters used.
	 */
	unsigned int getNumFiltersToUse(void) const;

	/**
	 * Sets the number of filters (RSVs) this RVM is currently using for classifying.
	 * If set to zero, all available filters will be used. If set higher than the number
	 * of available filters, then also all available filters will be used.
	 *
	 * @param[in] The number of filters to be used.
	 */
	void setNumFiltersToUse(const unsigned int numFilters);

	/**
	 * Creates a new RVM classifier from a Matlab file (.mat) containing the classifier and
	 * a second .mat file containing the thresholds.
	 *
	 * @param[in] classifierFilename The name of the file containing the RVM vectors and weights.
	 * @param[in] thresholdsFilename The name of the file containing the RVM thresholds.
	 * @return The newly created RVM classifier.
	 */
	static shared_ptr<RvmClassifier> loadMatlab(const string& classifierFilename, const string& thresholdsFilename);

	/**
	 * Creates a new RVM classifier from the parameters given in the ptree sub-tree. Passes the
	 * loading to the respective loading routine (at the moment, only Matlab files are supported).
	 *
	 * @param[in] subtree The subtree containing the config information for this classifier.
	 * @return The newly created RVM classifier.
	 */
	static shared_ptr<RvmClassifier> loadConfig(const ptree& subtree);


private:

	vector<Mat> supportVectors;				///< The support vectors (or here, RSVs).
	vector<vector<float>> coefficients;		///< The coefficients of the support vectors. Each step in the cascade has its own set of coefficients.
	int numFiltersToUse;					///< The number of filters to use out of the total number.
	vector<float> hierarchicalThresholds;	///< A classification threshold for each filter.

};

} /* namespace classification */
#endif /* RVMCLASSIFIER_HPP_ */
