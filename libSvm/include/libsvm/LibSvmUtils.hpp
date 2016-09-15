/*
 * LibSvmUtils.hpp
 *
 *  Created on: 12.09.2013
 *      Author: poschmann
 */

#ifndef LIBSVMUTILS_HPP_
#define LIBSVMUTILS_HPP_

#include "opencv2/core/core.hpp"
#include <unordered_map>
#include <memory>
#include <vector>

struct svm_node;
struct svm_parameter;
struct svm_problem;
struct svm_model;

namespace classification {
class Kernel;
} /* namespace classification */

namespace libsvm {

/**
 * Deleter of libSVM nodes also removes the cv::Mat representation of that node within a map.
 */
class NodeDeleter {
public:
	/**
	 * Constructs a new node deleter.
	 *
	 * @param[in] map The map the nodes should be removed from on deletion.
	 */
	NodeDeleter(std::unordered_map<const struct svm_node*, cv::Mat>& map);
	NodeDeleter(const NodeDeleter& other);
	NodeDeleter& operator=(const NodeDeleter& other);
	void operator()(struct svm_node *node) const;
private:
	std::unordered_map<const struct svm_node*, cv::Mat>& map; ///< The map the nodes should be removed from on deletion.
};

/**
 * Deleter of the libSVM parameter.
 */
class ParameterDeleter {
public:
	void operator()(struct svm_parameter *param) const;
};

/**
 * Deleter of the libSVM problem.
 */
class ProblemDeleter {
public:
	void operator()(struct svm_problem *problem) const;
};

/**
 * Deleter of the libSVM model.
 */
class ModelDeleter {
public:
	void operator()(struct svm_model *model) const;
};

/**
 * Utility class for libSVM with functions for creating nodes and computing SVM outputs. Usable via
 * composition or inheritance.
 */
class LibSvmUtils {
public:

	LibSvmUtils();

	virtual ~LibSvmUtils();

	/**
	 * @return Deleter of libSVM nodes that removes the node from the map.
	 */
	NodeDeleter getNodeDeleter() const;

	/**
	 * Creates a new libSVM node from the given feature vector data. Will store the vector in a map for later retrieval.
	 * The vectors is removed from the map on node deletion.
	 *
	 * @param[in] vector The feature vector.
	 * @return The newly created libSVM node.
	 */
	std::unique_ptr<struct svm_node[], NodeDeleter> createNode(const cv::Mat& vector) const;

	/**
	 * Retrieves the vector to the given libSVM node. Will have a look into the map of stored vectors first. If the vector
	 * is not contained within the map, it is created and stored within the map. Therefore this function should only be
	 * called with nodes that reside within a unique_ptr with a NodeDeleter to ensure that vectors are cleaned up.
	 *
	 * @param[in] node The libSVM node.
	 * @return The feature vector.
	 */
	cv::Mat getVector(const struct svm_node *node) const;

	/**
	 * Sets the kernel parameters for libSVM training.
	 *
	 * @param[in] kernel The kernel.
	 * @param[in] params The libSVM parameters.
	 */
	void setKernelParams(const classification::Kernel& kernel, struct svm_parameter *params) const;

	/**
	 * Computes the SVM output given a libSVM node.
	 *
	 * @param[in] model The libSVM model.
	 * @param[in] x The libSVM node.
	 * @return The SVM output value.
	 */
	double computeSvmOutput(struct svm_model *model, const struct svm_node *x) const;

	/**
	 * Extracts the support vectors from a libSVM model.
	 *
	 * @param[in] model The libSVM model.
	 * @return The support vectors.
	 */
	std::vector<cv::Mat> extractSupportVectors(struct svm_model *model) const;

	/**
	 * Extracts the coefficients from a libSVM model.
	 *
	 * @param[in] model The libSVM model.
	 * @return The coefficients.
	 */
	std::vector<float> extractCoefficients(struct svm_model *model) const;

	/**
	 * Extracts the bias from a libSVM model.
	 *
	 * @param[in] model The libSVM model.
	 * @return The bias.
	 */
	double extractBias(struct svm_model *model) const;

	/**
	 * Extracts parameter a from the logistic function that computes the probability
	 * p(x) = 1 / (1 + exp(a * x + b)) with x being the hyperplane distance.
	 *
	 * @param[in] model The libSVM model.
	 * @return The param a of the logistic function.
	 */
	double extractLogisticParamA(struct svm_model *model) const;

	/**
	 * Extracts parameter b from the logistic function that computes the probability
	 * p(x) = 1 / (1 + exp(a * x + b)) with x being the hyperplane distance.
	 *
	 * @param[in] model The libSVM model.
	 * @return The param b of the logistic function.
	 */
	double extractLogisticParamB(struct svm_model *model) const;

private:

	/**
	 * Fills a libSVM node with the data of a feature vector.
	 *
	 * @param[in,out] node The libSVM node.
	 * @param[in] vector The feature vector.
	 * @param[in] size The size of the data.
	 */
	template<class T>
	void fillNode(struct svm_node *node, const cv::Mat& vector, int size) const;

	/**
	 * Fills a vector with the data of a libSVM node.
	 *
	 * @param[in,out] vector The vector.
	 * @param[in] node The libSVM node.
	 * @param[in] size The size of the data.
	 */
	template<class T>
	void fillMat(cv::Mat& vector, const struct svm_node *node, int size) const;

	mutable int matRows;  ///< The row count of the support vector data.
	mutable int matCols;  ///< The column count of the support vector data.
	mutable int matType;  ///< The type of the support vector data.
	mutable int matDepth; ///< The depth of the support vector data.
	mutable int dimensions; ///< The amount of dimensions of the feature vectors.
	mutable std::unordered_map<const struct svm_node*, cv::Mat> node2example; ///< Maps libSVM nodes to the training examples they were created with.
	NodeDeleter nodeDeleter; ///< Deleter of libSVM nodes that removes the node from the map.
};

} /* namespace libsvm */


#endif /* LIBSVMUTILS_HPP_ */
