/*
 * regressors.hpp
 *
 *  Created on: 05.12.2014
 *      Author: Patrik Huber
 */
#pragma once

#ifndef REGRESSORS_HPP_
#define REGRESSORS_HPP_

#include "matserialisation.hpp"
#include "logging/LoggerFactory.hpp"

#include "opencv2/core/core.hpp"
#include "Eigen/Dense"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/serialization/vector.hpp"

#include <memory>
#include <chrono>

namespace superviseddescent {
	namespace v2 {

/**
 * Abstract base class for regressor-like learning
 * algorithms to be used with the SupervisedDescentOptimiser.
 *
 * Classes that implement this minimal set of functions
 * can be used with the SupervisedDescentOptimiser.
 */
class Regressor
{
public:
	virtual ~Regressor() {};

	/**
	 * Desc.
	 *
	 * @param[in] data Desc.
	 * @param[in] labels Desc.
	 * @return Todo.
	 */
	// or train()
	// Returns whether the learning was successful.
	// or maybe the prediction error? => no, for that case, we can use test() with the same data.
	virtual bool learn(cv::Mat data, cv::Mat labels) = 0;

	/**
	 * Desc.
	 *
	 * @param[in] data Desc.
	 * @param[in] labels Desc.
	 * @return Todo.
	 */
	// maybe, for testing:
	// return a results struct or something, like in tiny-cnn
	// Or atm the normalised LS residual
	virtual double test(cv::Mat data, cv::Mat labels) = 0;

	/**
	 * Desc.
	 *
	 * @param[in] values Desc.
	 * @return Todo.
	 */
	virtual cv::Mat predict(cv::Mat values) = 0;

	// overload for float, let's see if we can use it in this way
	// I think I ended up not using this => delete if so
	virtual float predict(float value) = 0;
};

class Regulariser
{
public:
	/**
	 * Todo: Description.
	 */
	enum class RegularisationType
	{
		Manual, ///< use lambda
		MatrixNorm, ///< use norm... optional lambda used as factor. If used, a suitable default is 0.5f.As mentioned by the authors of [SDM].
		EigenvalueThreshold ///< see description libEigen
	};

	Regulariser(RegularisationType regularisationType = RegularisationType::Manual, float param = 0.0f, bool regulariseLastRow = true) : regularisationType(regularisationType), lambda(param), regulariseLastRow(regulariseLastRow)
	{
	};

	// Returns a diagonal regularisation matrix with the same dimensions as the given data matrix.
	// Might use the data to calculate an automatic value for lambda.
	cv::Mat getMatrix(cv::Mat data, int numTrainingElements)
	{
		auto logger = logging::Loggers->getLogger("superviseddescent");
		// move regularisation to a separate function, so we can also test it separately. getRegulariser()?
		// Actually all that stuff can go in a class Regulariser
		switch (regularisationType)
		{
		case RegularisationType::Manual:
			// We just take lambda as it was given, no calculation necessary.
			break;
		case RegularisationType::MatrixNorm:
			// The given lambda is the factor we have to multiply the automatic value with
			lambda = lambda * static_cast<float>(cv::norm(data)) / static_cast<float>(numTrainingElements); // We divide by the number of images.
			// However, division by (AtA.rows * AtA.cols) might make more sense? Because this would be an approximation for the
			// RMS (eigenvalue? see sheet of paper, ev's of diag-matrix etc.), and thus our (conservative?) guess for a lambda that makes AtA invertible.
			break;
		case RegularisationType::EigenvalueThreshold:
			//lambda = calculateEigenvalueThreshold(AtA);
			lambda = 0.5f; // TODO COPY FROM OLD CODE
			break;
		default:
			break;
		}
		logger.debug("Lambda is: " + std::to_string(lambda));

		cv::Mat regulariser = cv::Mat::eye(data.rows, data.cols, CV_32FC1) * lambda; // always symmetric

		if (!regulariseLastRow) {
			regulariser.at<float>(regulariser.rows - 1, regulariser.cols - 1) = 0.0f; // no lambda for the bias
		}
		return regulariser;
	};

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & regularisationType;
		ar & lambda;
		ar & regulariseLastRow;
	}

private:
	RegularisationType regularisationType;
	float lambda; // param for RegularisationType. Can be lambda directly or a factor with which the lambda from MatrixNorm will be multiplied with.
	bool regulariseLastRow; // if your last row of data matrix is a bias (offset), then you might want to choose whether it should be regularised as well. Otherwise, just leave it to default (true).
};

class LinearRegressor : public Regressor
{

public:
	// The default should be the simplest case: Do not regularisation at all.
	// regulariseBias assumes the last row (col?) that will be given as training is a row to learn a bias b (affine component)
	// ==> If anywhere, learnBias should go to SDO class. Nothing changes in the regressor.
	// If you want to learn a bias term, add it as a column with [1; 1; ...; 1] to the data
	// A few of those params could go to the learn function?
	LinearRegressor(Regulariser regulariser = Regulariser()) : x(), regulariser(regulariser)
	{
	};

	// Returns if the learning was successful or not. I.e. in this case, if the matrix was invertible.
	bool learn(cv::Mat data, cv::Mat labels)
	{
		using cv::Mat;
		logging::Logger logger = logging::Loggers->getLogger("superviseddescent");
		std::chrono::time_point<std::chrono::system_clock> start, end;
		int elapsed_mseconds;

		Mat AtA = data.t() * data;

		Mat regularisationMatrix = regulariser.getMatrix(AtA, data.rows);
		
		// solve for x!
		Mat AtAReg = AtA + regularisationMatrix;
		if (!AtAReg.isContinuous()) {
			std::string msg("Matrix is not continuous. This should not happen as we allocate it directly.");
			logger.error(msg);
			throw std::runtime_error(msg);
		}
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> AtAReg_Eigen(AtAReg.ptr<float>(), AtAReg.rows, AtAReg.cols);
		std::chrono::time_point<std::chrono::system_clock> inverseTimeStart = std::chrono::system_clock::now();
		// Calculate the full-pivoting LU decomposition of the regularized AtA. Note: We could also try FullPivHouseholderQR if our system is non-minimal (i.e. there are more constraints than unknowns).
		Eigen::FullPivLU<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> luOfAtAReg(AtAReg_Eigen);
		// we could also print the smallest eigenvalue here, but that would take time to calculate (SelfAdjointEigenSolver, see above)
		float rankOfAtAReg = luOfAtAReg.rank(); // Todo: Use auto?
		logger.trace("Rank of the regularized AtA: " + std::to_string(rankOfAtAReg));
		if (luOfAtAReg.isInvertible()) {
			logger.debug("The regularized AtA is invertible.");
		}
		else {
			// Eigen will most likely return garbage here (according to their docu anyway). We have a few options:
			// - Increase lambda
			// - Calculate the pseudo-inverse. See: http://eigen.tuxfamily.org/index.php?title=FAQ#Is_there_a_method_to_compute_the_.28Moore-Penrose.29_pseudo_inverse_.3F
			std::string msg("The regularised AtA is not invertible. We continued learning, but Eigen calculates garbage in this case according to their documentation. (The rank is " + std::to_string(rankOfAtAReg) + ", full rank would be " + std::to_string(AtAReg_Eigen.rows()) + "). Increase lambda (or use the pseudo-inverse, which is not implemented yet).");
			logger.error(msg);
		}
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> AtARegInv_EigenFullLU = luOfAtAReg.inverse();
		//Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> AtARegInv_Eigen = AtAReg_Eigen.inverse(); // This would be the cheap variant (PartialPivotLU), but we can't check if the matrix is invertible.
		Mat AtARegInvFullLU(AtARegInv_EigenFullLU.rows(), AtARegInv_EigenFullLU.cols(), CV_32FC1, AtARegInv_EigenFullLU.data()); // create an OpenCV Mat header for the Eigen data
		std::chrono::time_point<std::chrono::system_clock> inverseTimeEnd = std::chrono::system_clock::now();
		elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(inverseTimeEnd - inverseTimeStart).count();
		logger.debug("Inverting the regularized AtA took " + std::to_string(elapsed_mseconds) + "ms.");
		//Mat AtARegInvOCV = AtAReg.inv(); // slow OpenCV inv() for comparison

		// Todo(1): Moving AtA by lambda should move the eigenvalues by lambda, however, it does not. It did however on an early test (with rand maybe?).
		//Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(AtAReg_Eigen);
		//std::cout << es.eigenvalues() << std::endl;

		Mat AtARegInvAt = AtARegInvFullLU * data.t(); // Todo: We could use luOfAtAReg.solve(b, x) instead of .inverse() and these lines.
		Mat AtARegInvAtb = AtARegInvAt * labels; // = x
		
		x = AtARegInvAtb; // store in member variable.
		
		return luOfAtAReg.isInvertible();
	};

	// Return the normalised least square residuals?
	// $\frac{\|\mathbf{x}_k-\mathbf{x^*}\|}{\|\mathbf{x^*}\|}$
	// $k$ is the current regressor step, i.e. this regressor only returns the residual for this step
	double test(cv::Mat data, cv::Mat labels)
	{
		cv::Mat predictions;
		for (int i = 0; i < data.rows; ++i) {
			cv::Mat prediction = predict(data.row(i));
			predictions.push_back(prediction);
		}
		return cv::norm(predictions, labels, cv::NORM_L2) / cv::norm(labels, cv::NORM_L2);
	};

	// values has to be a row vector. One row for every data point.
	// The input to this function is only 1 data point, i.e. one row-vector.
	cv::Mat predict(cv::Mat values)
	{
		cv::Mat prediction = values * x;
		return prediction;
	};

	float predict(float value)
	{
		if (x.rows != 1 || x.cols != 1) {
			throw std::runtime_error("Trying to predict a float value, but the used regressor doesn't have exactly one row and one column.");
		}
		float prediction = value * x.at<float>(0);
		return prediction;
	};

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & x;
		ar & regulariser;
	}
	
	cv::Mat x; // move back to private once finished
	// could rename to 'R' (as used in the SDM)
private:
	Regulariser regulariser;

	// Add serialisation
};

	} /* namespace v2 */
} /* namespace superviseddescent */
#endif /* REGRESSORS_HPP_ */
