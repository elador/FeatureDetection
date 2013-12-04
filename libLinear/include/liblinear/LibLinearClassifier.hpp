/*
 * LibLinearClassifier.hpp
 *
 *  Created on: 03.12.2013
 *      Author: poschmann
 */

#ifndef LIBLINEARCLASSIFIER_HPP_
#define LIBLINEARCLASSIFIER_HPP_

#include "classification/TrainableSvmClassifier.hpp"
#include "liblinear/LibLinearUtils.hpp"
#include "linear.h"
#include <string>

namespace classification {
class ExampleManagement;
}; /* namespace classification */

namespace liblinear {

/**
 * Trainable SVM classifier that uses libLinear for training.
 */
class LibLinearClassifier : public classification::TrainableSvmClassifier {
public:

	/**
	 * Constructs a new libLinear classifier.
	 *
	 * @param[in] c The constraints violation costs C.
	 * @param[in] bias Flag that indicates whether there should be a bias.
	 */
	explicit LibLinearClassifier(double c = 1, bool bias = false);

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

private:

	/**
	 * Trains the SVM using all of the stored training examples.
	 *
	 * @return True if this SVM was trained successfully and may be used, false otherwise.
	 */
	bool train();

	/**
	 * Creates libLinear nodes from training examples.
	 *
	 * @param[in] examples Training examples.
	 * @return Vector of libLinear nodes.
	 */
	std::vector<std::unique_ptr<struct feature_node[]>> createNodes(classification::ExampleManagement* examples);

	/**
	 * Creates the libLinear problem containing the training data.
	 *
	 * @param[in] positiveExamples Positive training examples.
	 * @param[in] negativeExamples Negative training examples.
	 * @param[in] staticNegativeExamples Static negative training examples.
	 * @param[in] dimensions The amount of dimensions of the feature vectors (without the bias).
	 * @return The libLinear problem.
	 */
	std::unique_ptr<struct problem, ProblemDeleter> createProblem(
			const std::vector<std::unique_ptr<struct feature_node[]>>& positiveExamples,
			const std::vector<std::unique_ptr<struct feature_node[]>>& negativeExamples,
			const std::vector<std::unique_ptr<struct feature_node[]>>& staticNegativeExamples,
			size_t dimensions);

	LibLinearUtils utils; ///< Utils for using libLinear.
	bool bias; ///< Flag that indicates whether there should be a bias.
	std::unique_ptr<struct parameter, ParameterDeleter> param; ///< The libLinear parameters.
	std::vector<std::unique_ptr<struct feature_node[]>> staticNegativeExamples; ///< The static negative training examples.
	std::unique_ptr<classification::ExampleManagement> positiveExamples; ///< Storage of positive training examples.
	std::unique_ptr<classification::ExampleManagement> negativeExamples; ///< Storage of negative training examples.
};

} /* namespace liblinear */
#endif /* LIBLINEARCLASSIFIER_HPP_ */
