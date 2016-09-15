/*
 * LibSvmClassifier.hpp
 *
 *  Created on: 21.11.2013
 *      Author: poschmann
 */

#ifndef LIBSVMCLASSIFIER_HPP_
#define LIBSVMCLASSIFIER_HPP_

#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/TrainableSvmClassifier.hpp"
#include "libsvm/LibSvmUtils.hpp"
#include <string>

namespace classification {
class ExampleManagement;
}; /* namespace classification */

namespace libsvm {

/**
 * Trainable SVM classifier that uses libSVM for training.
 */
class LibSvmClassifier : public classification::TrainableSvmClassifier {
public:

	/**
	 * Creates a new one-class support vector machine that will be trained with libSVM.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] nu The nu.
	 */
	static std::shared_ptr<LibSvmClassifier> createOneClassSvm(std::shared_ptr<classification::Kernel> kernel, double nu = 1) {
		return std::make_shared<LibSvmClassifier>(kernel, nu, true, false);
	}

	/**
	 * Creates a new one-class support vector machine that will be trained with libSVM.
	 *
	 * @param[in] svm The actual SVM.
	 * @param[in] nu The nu.
	 */
	static std::shared_ptr<LibSvmClassifier> createOneClassSvm(std::shared_ptr<classification::SvmClassifier> svm, double nu = 1) {
		return std::make_shared<LibSvmClassifier>(svm, nu, true, false);
	}

	/**
	 * Creates a new binary support vector machine that will be trained with libSVM.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] c The C.
	 * @param[in] compensateImbalance Flag that indicates whether to adjust class weights to compensate for unbalanced data.
	 * @param[in] probabilistic Flag that indicates whether to compute logistic parameters for probabilistic output.
	 */
	static std::shared_ptr<LibSvmClassifier> createBinarySvm(
			std::shared_ptr<classification::Kernel> kernel, double c = 1, bool compensateImbalance = false, bool probabilistic = false) {
		return std::make_shared<LibSvmClassifier>(kernel, c, false, compensateImbalance, probabilistic);
	}

	/**
	 * Creates a new binary support vector machine that will be trained with libSVM.
	 *
	 * @param[in] svm The actual SVM.
	 * @param[in] c The C.
	 * @param[in] compensateImbalance Flag that indicates whether to adjust class weights to compensate for unbalanced data.
	 * @param[in] probabilistic Flag that indicates whether to compute logistic parameters for probabilistic output.
	 */
	static std::shared_ptr<LibSvmClassifier> createBinarySvm(
			std::shared_ptr<classification::SvmClassifier> svm, double c = 1, bool compensateImbalance = false, bool probabilistic = false) {
		return std::make_shared<LibSvmClassifier>(svm, c, false, compensateImbalance, probabilistic);
	}

	/**
	 * Constructs a new libSVM classifier.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] cnu The parameter C in case of an ordinary SVM, nu in case of a one-class SVM.
	 * @param[in] oneClass Flag that indicates whether a one-class SVM should be trained.
	 * @param[in] compensateImbalance Flag that indicates whether to adjust class weights to compensate for unbalanced data.
	 * @param[in] probabilistic Flag that indicates whether to compute logistic parameters for probabilistic output.
	 */
	LibSvmClassifier(std::shared_ptr<classification::Kernel> kernel, double cnu, bool oneClass,
			bool compensateImbalance = false, bool probabilistic = false);

	/**
	 * Constructs a new libSVM classifier.
	 *
	 * @param[in] svm The actual SVM.
	 * @param[in] cnu The parameter C in case of an ordinary SVM, nu in case of a one-class SVM.
	 * @param[in] oneClass Flag that indicates whether a one-class SVM should be trained.
	 * @param[in] compensateImbalance Flag that indicates whether to adjust class weights to compensate for unbalanced data.
	 * @param[in] probabilistic Flag that indicates whether to compute logistic parameters for probabilistic output.
	 */
	LibSvmClassifier(std::shared_ptr<classification::SvmClassifier> svm, double cnu, bool oneClass,
			bool compensateImbalance = false, bool probabilistic = false);

public:

	/**
	 * Loads static negative training examples from a file.
	 *
	 * @param[in] negativesFilename The name of the file containing the static negative training examples.
	 * @param[in] maxNegatives The amount of static negative training examples to use.
	 * @param[in] scale The factor for scaling the data after loading.
	 */
	void loadStaticNegatives(const std::string& negativesFilename, int maxNegatives, double scale = 1);

	bool retrain(const std::vector<cv::Mat>& newPositiveExamples, const std::vector<cv::Mat>& newNegativeExamples);

	void reset();

	/**
	 * @param[in] positiveExamples Storage of positive training examples.
	 */
	void setPositiveExampleManagement(std::unique_ptr<classification::ExampleManagement> positiveExamples) {
		this->positiveExamples = move(positiveExamples);
	}

	/**
	 * @param[in] negativeExamples Storage of negative training examples.
	 */
	void setNegativeExampleManagement(std::unique_ptr<classification::ExampleManagement> negativeExamples) {
		this->negativeExamples = move(negativeExamples);
	}

	/**
	 * @return The actual probabilistic SVM classifier.
	 */
	std::shared_ptr<classification::ProbabilisticSvmClassifier> getProbabilisticSvm() {
		return probabilisticSvm;
	}

	/**
	 * @return The actual probabilistic SVM classifier.
	 */
	const std::shared_ptr<classification::ProbabilisticSvmClassifier> getProbabilisticSvm() const {
		return probabilisticSvm;
	}

private:

	/**
	 * Creates the libSVM parameters.
	 *
	 * @param[in] kernel The kernel function.
	 * @param[in] cnu The parameter C in case of an ordinary SVM, nu in case of a one-class SVM.
	 * @param[in] oneClass Flag that indicates whether a one-class SVM should be trained.
	 */
	void createParameters(std::shared_ptr<classification::Kernel> kernel, double cnu, bool oneClass);

	/**
	 * Trains the SVM using all of the stored training examples.
	 *
	 * @return True if this SVM was trained successfully and may be used, false otherwise.
	 */
	bool train();

	/**
	 * Creates libSVM nodes from training examples.
	 *
	 * @param[in] examples Training examples.
	 * @return Vector of libSVM nodes.
	 */
	std::vector<std::unique_ptr<struct svm_node[], NodeDeleter>> createNodes(classification::ExampleManagement* examples);

	/**
	 * Creates the libSVM problem containing the training data.
	 *
	 * @param[in] positiveExamples Positive training examples.
	 * @param[in] negativeExamples Negative training examples.
	 * @param[in] staticNegativeExamples Static negative training examples.
	 * @return The libSVM problem.
	 */
	std::unique_ptr<struct svm_problem, ProblemDeleter> createProblem(
			const std::vector<std::unique_ptr<struct svm_node[], NodeDeleter>>& positiveExamples,
			const std::vector<std::unique_ptr<struct svm_node[], NodeDeleter>>& negativeExamples,
			const std::vector<std::unique_ptr<struct svm_node[], NodeDeleter>>& staticNegativeExamples);

	bool compensateImbalance; ///< Flag that indicates whether to adjust class weights to compensate for unbalanced data.
	bool probabilistic; ///< Flag that indicates whether to compute logistic parameters for probabilistic output.
	std::shared_ptr<classification::ProbabilisticSvmClassifier> probabilisticSvm; ///< The actual probabilistic SVM classifier.
	LibSvmUtils utils; ///< Utils for using libSVM.
	std::unique_ptr<struct svm_parameter, ParameterDeleter> param; ///< The libSVM parameters.
	std::unique_ptr<classification::ExampleManagement> positiveExamples; ///< Storage of positive training examples.
	std::unique_ptr<classification::ExampleManagement> negativeExamples; ///< Storage of negative training examples.
	std::vector<std::unique_ptr<struct svm_node[], NodeDeleter>> staticNegativeExamples; ///< The static negative training examples.
};

} /* namespace libsvm */
#endif /* LIBSVMCLASSIFIER_HPP_ */
