/*
 * superviseddescent.hpp.hpp
 *
 *  Created on: 14.11.2014
 *      Author: Patrik Huber
 */

#pragma once

#ifndef SUPERVISEDDESCENT_HPP_
#define SUPERVISEDDESCENT_HPP_

#include "logging/LoggerFactory.hpp"

#include "opencv2/core/core.hpp"
#include "Eigen/Dense"

#include "boost/lexical_cast.hpp"

#include <memory>
#include <chrono>

using logging::Logger; // Todo: Move to cpp as soon as finished
using logging::LoggerFactory;
using cv::Mat;
using boost::lexical_cast;
using std::string;

namespace superviseddescent {
	namespace v2 {

/**
 * Stuff for supervised descent v2
 * Will be moved to separate files / cpp files once finished
 */

class Regressor
{
public:
	virtual ~Regressor() {};

	// or train()
	// return? maybe the prediction error?
	virtual void learn(cv::Mat data, cv::Mat labels) = 0;

	// maybe, for testing:
	// return a results struct or something, like in tiny-cnn
	// Or atm the normalised LS residual
	virtual double test(cv::Mat data, cv::Mat labels) = 0;

	virtual cv::Mat predict(cv::Mat values) = 0;
};

class LinearRegressor : public Regressor
{

public:
	/**
	 * Todo: Description.
	 */
	enum class RegularisationType
	{
		Manual, ///< use lambda
		Automatic, ///< use norm... optional lambda used as factor
		EigenvalueThreshold ///< see description libEigen
	};

	LinearRegressor(RegularisationType regularisationType = RegularisationType::Automatic, float lambda = 0.5f, bool regulariseAffineComponent = true) : x(), regularisationType(regularisationType), lambda(lambda), regulariseAffineComponent(regulariseAffineComponent)
	{

	};

	void learn(cv::Mat data, cv::Mat labels)
	{
		Logger logger = Loggers->getLogger("superviseddescent");
		std::chrono::time_point<std::chrono::system_clock> start, end;
		int elapsed_mseconds;

		Mat AtA = data.t() * data;

		switch (regularisationType)
		{
		case RegularisationType::Manual:
			// We just take lambda as it was given, no calculation necessary.
			break;
		case RegularisationType::Automatic:
			// The given lambda is the factor we have to multiply the automatic value with
			lambda = lambda * static_cast<float>(cv::norm(AtA)) / static_cast<float>(data.rows); // We divide by the number of images.
			// However, division by (AtA.rows * AtA.cols) might make more sense? Because this would be an approximation for the
			// RMS (eigenvalue? see sheet of paper, ev's of diag-matrix etc.), and thus our (conservative?) guess for a lambda that makes AtA invertible.
			break;
		case RegularisationType::EigenvalueThreshold:
		{
			std::chrono::time_point<std::chrono::system_clock> eigenTimeStart = std::chrono::system_clock::now();
			//lambda = calculateEigenvalueThreshold(AtA);
			lambda = 0.5f; // TODO COPY FROM OLD CODE
			std::chrono::time_point<std::chrono::system_clock> eigenTimeEnd = std::chrono::system_clock::now();
			elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(eigenTimeEnd - eigenTimeStart).count();
			logger.debug("Automatic calculation of lambda for AtA took " + lexical_cast<string>(elapsed_mseconds)+"ms.");
		}
			break;
		default:
			break;
		}
		logger.debug("Setting lambda to: " + lexical_cast<string>(lambda));

		Mat regulariser = Mat::eye(AtA.rows, AtA.rows, CV_32FC1) * lambda;
		if (!regulariseAffineComponent) {
			// Note: The following line breaks if we enter 1-dimensional data
			// Update: No, it doesn't crash, it points to (0, 0) then. Has the wrong effect though.
			regulariser.at<float>(regulariser.rows - 1, regulariser.cols - 1) = 0.0f; // no lambda for the bias
		}
		// solve for x!
		Mat AtAReg = AtA + regulariser;
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
		logger.trace("Rank of the regularized AtA: " + lexical_cast<string>(rankOfAtAReg));
		if (luOfAtAReg.isInvertible()) {
			logger.debug("The regularized AtA is invertible.");
		}
		else {
			// Eigen will most likely return garbage here (according to their docu anyway). We have a few options:
			// - Increase lambda
			// - Calculate the pseudo-inverse. See: http://eigen.tuxfamily.org/index.php?title=FAQ#Is_there_a_method_to_compute_the_.28Moore-Penrose.29_pseudo_inverse_.3F
			string msg("The regularized AtA is not invertible (its rank is " + lexical_cast<string>(rankOfAtAReg)+", full rank would be " + lexical_cast<string>(AtAReg_Eigen.rows()) + "). Increase lambda (or use the pseudo-inverse, which is not implemented yet).");
			logger.error(msg);
#ifndef _DEBUG
			//throw std::runtime_error(msg); // Don't throw while debugging. Makes debugging with small amounts of data possible.
#endif
		}
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> AtARegInv_EigenFullLU = luOfAtAReg.inverse();
		//Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> AtARegInv_Eigen = AtAReg_Eigen.inverse(); // This would be the cheap variant (PartialPivotLU), but we can't check if the matrix is invertible.
		Mat AtARegInvFullLU(AtARegInv_EigenFullLU.rows(), AtARegInv_EigenFullLU.cols(), CV_32FC1, AtARegInv_EigenFullLU.data()); // create an OpenCV Mat header for the Eigen data
		std::chrono::time_point<std::chrono::system_clock> inverseTimeEnd = std::chrono::system_clock::now();
		elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(inverseTimeEnd - inverseTimeStart).count();
		logger.debug("Inverting the regularized AtA took " + lexical_cast<string>(elapsed_mseconds) + "ms.");
		//Mat AtARegInvOCV = AtAReg.inv(); // slow OpenCV inv() for comparison

		// Todo(1): Moving AtA by lambda should move the eigenvalues by lambda, however, it does not. It did however on an early test (with rand maybe?).
		//Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(AtAReg_Eigen);
		//std::cout << es.eigenvalues() << std::endl;

		Mat AtARegInvAt = AtARegInvFullLU * data.t(); // Todo: We could use luOfAtAReg.solve(b, x) instead of .inverse() and these lines.
		Mat AtARegInvAtb = AtARegInvAt * labels; // = x
		x = AtARegInvAtb; // store in member variable.
		return; // Could we return something useful?
		// Maybe isInvertible. This can happen in normal control flow, so prefer this to exceptions?
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
		cv::Mat difference = predictions - labels;
		double differenceNorm = cv::norm(predictions, labels, cv::NORM_L2);
		double residual = differenceNorm / cv::norm(labels, cv::NORM_L2); // TODO Check the paper - x* is the ground truth, right?
		return residual;
	};

	// values has to be a row vector. One row for every data point.
	// The input to this function is only 1 data point, i.e. one row-vector.
	cv::Mat predict(cv::Mat values)
	{
		cv::Mat prediction = values.mul(x); // .dot?
		return prediction;
	};
	
	cv::Mat x; // move back to private once finished
private:
	
	RegularisationType regularisationType;
	float lambda;
	bool regulariseAffineComponent;

	// Add serialisation
};




// or... LossFunction, CostFunction, Evaluator, ...?
// No inheritance, so the user can plug in anything without inheriting.
class TrivialEvaluationFunction
{

};

// template blabla.. requires Evaluator with function eval(Mat, blabla)
// or... Learner? SDMLearner? Just SDM? SupervDescOpt? SDMTraining?
// This at the moment handles the case of _known_ y (target) values.
class SupervisedDescentOptimiser
{
public:
	SupervisedDescentOptimiser(std::vector<std::shared_ptr<Regressor>> regressors) : regressors(regressors)
	{
	};

	// Hmm in case of LM / 3DMM, we might need to give more info than this? How to handle that?
	// Yea, some optimisers need access to the image data to optimise. Think about where this belongs.
	// The input to this will be something different (e.g. only images? a templated class?)
	template<class H>
	void train(cv::Mat x, cv::Mat y, cv::Mat x0, H h)
	{
		// data = x = the parameters we want to learn, ground truth labels for them = y. x0 = c = initialisation
		// Simple experiments with sin etc.
		auto logger = Loggers->getLogger("superviseddescent");
		Mat currentX = x0;
		std::cout << x0 << std::endl;
		for (auto&& regressor : regressors) {
			Mat features;
			for (int i = 0; i < currentX.rows; ++i) {
				features.push_back(h(currentX.at<float>(i)));
			}
			Mat insideRegressor = features - y; // Todo: Find better name ;-)
			// We got $\sum\|x_*^i - x_k^i + R_k(h(x_k^i)-y^i)\|_2^2 $
			// That is: $-b + Ax = 0$, $Ax = b$, $x = A^-1 * b$
			// Thus: $b = x_k^i - x_*^i$. 
			// This is correct! It's the other way round than in the CVPR 2013 paper though. However,
			// it cancels out, as we later subtract the learned direction, while in the 2013 paper they add it.
			Mat b = currentX - x; // correct
			
			// 1) Extract features where necessary
			// 2) Learn using that data
			regressor->learn(insideRegressor, b);
			auto tmp = dynamic_cast<LinearRegressor*>(regressor.get());
			logger.info(std::to_string(tmp->x.at<float>(0)));
			// 3) If not finished, 
			//    apply learned regressor, set data = newData or rather calculate new delta
			//    use it to learn the next regressor in next loop iter
			Mat x_k;
			// Mat x_k = currentX - tmp->x.at<float>(0) * (h(currentX) - y);
			for (int i = 0; i < currentX.rows; ++i) {
				x_k.push_back(currentX.at<float>(i) - tmp->x.at<float>(0) * (h(currentX.at<float>(i)) - y.at<float>(i))); // minus here is correct. CVPR paper got a '+'. See above for reason why.
			}
			currentX = x_k;
			std::cout << currentX << std::endl;
			double differenceNorm = cv::norm(x_k, x, cv::NORM_L2);
			double residual = differenceNorm / cv::norm(x, cv::NORM_L2);
			std::cout << "normalised least square residual: " << residual << std::endl;
		}
	};

	// x_groundtruth will only be used to calculate the residual!
	template<class H>
	void test(cv::Mat x_groundtruth, cv::Mat y, cv::Mat x0, H h)
	{
		auto logger = Loggers->getLogger("superviseddescent");
		Mat currentX = x0;
		std::cout << x0 << std::endl;
		for (auto&& regressor : regressors) {
			Mat features;
			for (int i = 0; i < currentX.rows; ++i) {
				features.push_back(h(currentX.at<float>(i)));
			}
			Mat insideRegressor = features - y; // Todo: Find better name ;-)

			auto tmp = dynamic_cast<LinearRegressor*>(regressor.get());
			logger.info(std::to_string(tmp->x.at<float>(0)));

			Mat x_k;
			// Mat x_k = currentX - tmp->x.at<float>(0) * (h(currentX) - y);
			for (int i = 0; i < currentX.rows; ++i) {
				x_k.push_back(currentX.at<float>(i) -tmp->x.at<float>(0) * (h(currentX.at<float>(i)) - y.at<float>(i)));
			}
			currentX = x_k;
			std::cout << currentX << std::endl;
			double differenceNorm = cv::norm(x_k, x_groundtruth, cv::NORM_L2);
			double residual = differenceNorm / cv::norm(x_groundtruth, cv::NORM_L2);
			std::cout << "normalised least square residual: " << residual << std::endl;

			//double residual = regressor->test(x0, labels); // Use this instead!
			// do some accumulation? printing?
			logger.info("Residual: " + std::to_string(residual));
		}
	};

private:
	// If we don't want to allow different regressor types, we
	// could make the Regressor a template parameter
	std::vector<std::shared_ptr<Regressor>> regressors; // Switch to shared_ptr, but VS2013 has a problem with it. VS2015 is fine.
	// I think it would be better to delete this ('CascadedRegressor') and store a
	// vector<R> instead in Optimiser, because multiple levels
	// may require different features and different normalisations (i.e. IED)?
	// Yep, try this first. But it has the disadvantage that we need to remove
	// this template parameter from Optimiser and use "standard" inheritance
	// instead. Same as in tiny-cnn.
	// The right approach seems definitely to put it in the Optimiser.
	// Because only he can now about the feature extraction required for each
	// regressor step, and the scales etc.

	//E evaluator;
};


/**
 * This implements ... todo.
 *
 * Usage example:
 * @code
 * auto testexp = [](float value) { return std::exp(value); };
 * auto testinv = [](float value) { return 1.0f / value; };
 * auto testpow = [](float value) { return std::pow(value, 2); };
 * float r_exp = 0.115f;
 * float r_inv = -7.0f;
 * float r_pow = 0.221;
 *
 * int dims = 31; Mat x_0_tr(dims, 1, CV_32FC1); // exp: [-2:0.2:4]
 * //int dims = 26; Mat x_0_tr(dims, 1, CV_32FC1); // inv: [1:0.2:6]
 * //int dims = 31; Mat x_0_tr(dims, 1, CV_32FC1); // pow: [0:0.2:6]
 * {
 *	vector<float> values(dims);
 * 	strided_iota(std::begin(values), std::next(std::begin(values), dims), -2.0f, 0.2f); // exp
 * 	//strided_iota(std::begin(values), std::next(std::begin(values), dims), 1.0f, 0.2f); // inv
 * 	//strided_iota(std::begin(values), std::next(std::begin(values), dims), 0.0f, 0.2f); // pow
 * 	x_0_tr = Mat(values, true);
 * }
 * Mat y_tr(31, 1, CV_32FC1); // exp: all = exp(1) = 2.7...;
 * //Mat y_tr(dims, 1, CV_32FC1); // inv: all = 0.286;
 * //Mat y_tr(31, 1, CV_32FC1); // pow: all = 9;
 * {
 * 	vector<float> values(dims);
 * 	std::fill(begin(values), end(values), std::exp(1.0f));
 * 	y_tr = Mat(values, true);
 * }
 *
 * v2::GenericDM1D g;
 * g.train(x_0_tr, y_tr, r_exp, testexp);
 * @endcode
 */
class GenericDM1D
{
public:
	template<typename H>
	void train(cv::Mat data, cv::Mat labels, float genericDescentMap, H h)
	{
		auto logger = Loggers->getLogger("superviseddescent");

		logger.info("r is: " + std::to_string(genericDescentMap));
		
		// For all training example, do the update step for 25 iterations.
		cv::Mat x_0 = data;
		cv::Mat x_prev = x_0;
		std::stringstream out;
		out << x_0 << std::endl;
		logger.info(out.str());
		for (int i = 1; i <= 25; ++i) {
			// Update x_0, the starting positions, using the learned regressor:
			//float x_next_0 = x_prev_0 - r.x.at<float>(0, 0) * (h(x_prev_0) - (labels.at<float>(0)));
			cv::Mat x_next(x_prev.rows, 1, CV_32FC1);
			for (int i = 0; i < x_prev.rows; ++i) {
				auto thisDelta = h(x_prev.at<float>(i)) - labels.at<float>(i);
				auto thisDeltaRegressed = genericDescentMap * thisDelta;
				x_next.at<float>(i) = x_prev.at<float>(i) -thisDeltaRegressed;
			}
			std::stringstream outIter;
			outIter << x_next << std::endl;
			logger.info(outIter.str());
			x_prev = x_next;
		}
	};
};

	} /* namespace v2 */
} /* namespace superviseddescent */
#endif /* SUPERVISEDDESCENT_HPP_ */
