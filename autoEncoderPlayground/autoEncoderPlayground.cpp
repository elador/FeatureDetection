/*
 * autoEncoderPlayground.cpp
 *
 *  Created on: 11.06.2014
 *      Author: Patrik Huber
 *
 * Example:
 * -
 *   
 */

// For memory leak debugging: http://msdn.microsoft.com/en-us/library/x98tx3cf(v=VS.100).aspx
//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#ifdef WIN32
	#include <SDKDDKVer.h>
#endif

/*	// There's a bug in boost/optional.hpp that prevents us from using the debug-crt with it
	// in debug mode in windows. It works in release mode, but as we need debugging, let's
	// disable the windows-memory debugging for now.
#ifdef WIN32
	#include <crtdbg.h>
#endif

#ifdef _DEBUG
	#ifndef DBG_NEW
		#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
		#define new DBG_NEW
	#endif
#endif  // _DEBUG
*/

#include <chrono>
#include <memory>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "opencv2/core/optim.hpp" // OpenCV 3.0.0 alpha

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/lexical_cast.hpp"

#include "Eigen/Dense"

#include "lbfgs.h"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using cv::Mat;
using cv::Point2f;
using cv::Vec3f;
using cv::Scalar;
using boost::filesystem::path;
using boost::lexical_cast;
using std::cout;
using std::endl;
using std::make_shared;
using std::string;
using std::vector;

std::pair<float, cv::Mat> sparseAutoencoderCost(cv::Mat theta, int visibleSize, int hiddenSize, float lambda, float sparsityParam, float beta, cv::Mat data);

static int progress(
	void *instance,
	const lbfgsfloatval_t *theta,
	const lbfgsfloatval_t *grad,
	const lbfgsfloatval_t cost,
	const lbfgsfloatval_t normTheta,
	const lbfgsfloatval_t normGrad,
	const lbfgsfloatval_t step,
	int nparam,
	int niter,
	int ls
	)
{
	cout << "Iteration: " << niter << "		cost: " << cost
		<< "	step: " << step << endl;
	return 0;
}

struct instance
{
	int visibleSize;
	int hiddenSize;
	double lambda;
	double sparsityParam;
	double beta;
	Mat data;
};

lbfgsfloatval_t sparseAECost(void* netParam, const lbfgsfloatval_t *ptheta, lbfgsfloatval_t *grad, const int n, const lbfgsfloatval_t step) {
	instance* pStruct = (instance*)(netParam);
	int hiddenSize = pStruct->hiddenSize;
	int visibleSize = pStruct->visibleSize;
	double lambda = pStruct->lambda;
	double beta = pStruct->beta;
	double sparsityParam = pStruct->sparsityParam;
	Mat data = pStruct->data;
	Mat theta(25 * 64 * 2 + 64 + 25, 1, CV_64FC1, (void*)ptheta);

	float cost;
	Mat gradret;
	std::tie(cost, gradret) = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data);

	Mat tmp = gradret.clone();
	if (!tmp.isContinuous()) {
		throw("ERR!");
	}
	//grad = tmp.ptr<double>(0);
	memcpy(grad, tmp.ptr<double>(0), n * sizeof(double));
	return cost;
};

class AECost : public cv::MinProblemSolver::Function {
public:

	AECost(int visibleSize, int hiddenSize, float lambda, float sparsityParam, float beta, cv::Mat data) : visibleSize(visibleSize), hiddenSize(hiddenSize), lambda(lambda), sparsityParam(sparsityParam), beta(beta) {
		this->data = data;
	};

	double calc(const double* x) const {
		//return x[0] * x[0] + x[1] * x[1]; 
		Mat theta(25 * 64 * 2 + 64 + 25, 1, CV_64FC1, (void*)x);
		float cost;
		Mat grad;
		std::tie(cost, grad) = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data);
		cout << "calc iter; cost: " << cost << endl;
		return cost;
	};

	// for Conjugate Gradient
	void getGradient(const double* x, double* grad) {
		Mat theta(25 * 64 * 2 + 64 + 25, 1, CV_64FC1, (void*)x);
		float cost;
		Mat gradret;
		std::tie(cost, gradret) = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data);
		cout << "grad iter; cost: " << cost << endl;
		Mat tmp = gradret.clone();
		if (!tmp.isContinuous()) {
			throw("ERR!");
		}
		//grad = tmp.ptr<double>(0);
		memcpy(grad, tmp.ptr<double>(0), theta.rows * sizeof(double));
	};

	int visibleSize;
	int hiddenSize;
	float lambda;
	float sparsityParam;
	float beta;
	cv::Mat data;
};


#include <random>
using cv::Rect;
std::vector<cv::Mat> generatePatches(std::vector<cv::Mat> images, int patchsize, int numPatches)
{
	if (images.size() == 0) {
		return vector<Mat>();
	}

	vector<Mat> patches;

	int imageWidth = images[0].cols;
	int imageHeight = images[0].rows;

	std::random_device rd; ///< Used for seeding the PRNG (mt19937)
	std::mt19937 engine(rd()); ///< A Mersenne twister mt19937 engine
	std::uniform_int_distribution<> randomImage(0, static_cast<int>(images.size()) - 1);
	std::uniform_int_distribution<> randomPointX(0, imageWidth - patchsize - 1); // maybe -1 or 0 or +1?
	std::uniform_int_distribution<> randomPointY(0, imageHeight - patchsize -1);

	for (auto i = 0; i < numPatches; ++i) {
		int randomIndex = randomImage(engine);
		Rect patchRegion;
		patchRegion.x = randomPointX(engine);
		patchRegion.y = randomPointY(engine);
		patchRegion.width = patchsize;
		patchRegion.height = patchsize;
		patches.emplace_back(images[randomIndex](patchRegion).clone()); // Yep, we really want to deep copy the data here.
	}
	return patches;
}

std::vector<cv::Mat> normalizeData(std::vector<cv::Mat> features)
{
	if (features.size() == 0) {
		return vector<Mat>();
	}
	// Squash data to[0.1, 0.9] since we use sigmoid as the activation
	// function in the output layer

	// subtract mean from each patch individually:
	// Remove DC(mean of images).
	//patches = bsxfun(@minus, patches, mean(patches));
	for (auto&& f : features) {
		cv::Scalar mean = cv::mean(f);
		f.convertTo(f, CV_32FC1);
		f = f - mean; // it's all done in-place, data is never copied. Good!
	}

	// Truncate to + / -3 standard deviations and scale to - 1 to 1. stddev of ALL patches together.
	//pstd = 3 * std(patches(:));
	Mat allPatches(features[0].rows * features[0].cols, static_cast<int>(features.size()), CV_32FC1); // each column is 1 patch
	for (auto i = 0; i < features.size(); ++i) {
		Mat reshaped = features[i].reshape(1, features[0].rows * features[0].cols);
		reshaped.copyTo(allPatches.col(i));
	}
	cv::Scalar stddev;
	cv::Scalar mean;
	cv::meanStdDev(allPatches, mean, stddev);
	float pstd = 3.0f * static_cast<float>(stddev[0]);
	//patches = max(min(patches, pstd), -pstd) / pstd; // % truncate each element individually
	allPatches = cv::max(cv::min(allPatches, pstd), -pstd) / pstd;
	// Rescale from[-1, 1] to[0.1, 0.9]
	allPatches = (allPatches + 1.0f) * 0.4f + 0.1f;

	vector<Mat> normalizedPatches;
	for (auto i = 0; i < allPatches.cols; ++i) {
		normalizedPatches.emplace_back(allPatches.col(i));
	}
	return normalizedPatches;
}

cv::Mat initializeParameters(int hiddenSize, int visibleSize)
{
	// Initialize parameters randomly based on layer sizes.
	float r = sqrt(6.0f) / static_cast<float>(sqrt(hiddenSize + visibleSize + 1));   // we'll choose weights uniformly from the interval [-r, r]
	std::random_device rd; ///< Used for seeding the PRNG (mt19937)
	std::mt19937 engine(rd()); ///< A Mersenne twister mt19937 engine
	std::uniform_real_distribution<float> randomNumber(-r, r);
	
	Mat W1(hiddenSize, visibleSize, CV_32FC1);
	for (auto r = 0; r < W1.rows; ++r) {
		for (auto c = 0; c < W1.cols; ++c) {
			W1.at<float>(r, c) = randomNumber(engine);
		}
	}
	Mat W2(visibleSize, hiddenSize, CV_32FC1);
	for (auto r = 0; r < W2.rows; ++r) {
		for (auto c = 0; c < W2.cols; ++c) {
			W2.at<float>(r, c) = randomNumber(engine);
		}
	}

	Mat b1(hiddenSize, 1, CV_32FC1, 0.0f);
	Mat b2(visibleSize, 1, CV_32FC1, 0.0f);

	// Convert weights and bias gradients to the vector form.
	// This step will "unroll" (flatten and concatenate together) all
	// your parameters into a vector, which can then be used with minFunc.
	//theta = [W1(:); W2(:); b1(:); b2(:)];
	//Mat theta(W1.rows * W1.cols + W2.rows * W2.cols + b1.rows + b2.rows, 1, CV_32FC1);
	Mat theta;
	cv::vconcat(W1.reshape(1, hiddenSize * visibleSize), W2.reshape(1, hiddenSize * visibleSize), theta);
	cv::vconcat(theta, b1, theta);
	cv::vconcat(theta, b2, theta);
	return theta;
}

template<class T>
T sigmoid(T val) {
	return 1 / (1 + std::exp(-val));
};

// element-wise sigmoid
// for CV32FC1 only!
template<>
cv::Mat sigmoid<cv::Mat>(cv::Mat val)
{
	if (val.type() != CV_64FC1) { // normally 32, but for cv Simplex, 64... tmp...
		throw std::runtime_error("Not CV_32FC1.");
	}
	Mat result(val.rows, val.cols, val.type());
	for (decltype(val.rows) r = 0; r < val.rows; ++r) {
		for (decltype(val.cols) c = 0; c < val.cols; ++c) {
			// Not sure how we can make the .at<float> call more generic
			// because the type is encoded in CV_32FC1.
			result.at<double>(r, c) = sigmoid(val.at<double>(r, c));
		}
	}
	return result;
}

// J is the cost function
// theta is the parameter vector we want to optimise (?)
template<class F>
cv::Mat computeNumericalGradient(F J, cv::Mat theta)
{
	Mat numgrad = Mat::zeros(theta.rows, theta.cols, theta.type()); // will be e.g. 3000 x 1 normally. Make sure it is, we will fail later otherwise?
	float epsilon = 0.0001f; // 1e-4;
	int n = theta.rows;
	Mat E = Mat::eye(n, n, CV_32FC1);

	for (int i = 0; i < n; ++i) {
		Mat delta = E.col(i) * epsilon;
		// for every parameter in theta, we compute the  numerical gradient
		// so basically for every (value +- delta), we compute the whole
		// neuralnet and its cost.
		// .first because we want the cost (it returns a pair(cost, grad))
		numgrad.at<float>(i) = (J(theta + delta).first - J(theta - delta).first) / (2.0f * epsilon);
	}

	return numgrad;
}

// visibleSize: the number of input units (probably 64)
// hiddenSize : the number of hidden units (probably 25)
// lambda : weight decay parameter
// sparsityParam : The desired average activation for the hidden units(denoted in the lecture
//                           notes by the greek alphabet rho, which looks like a lower - case "p").
// beta : weight of sparsity penalty term
// data : Our 64x10000 matrix containing the training data.So, data(:, i) is the i - th training example.

// The input theta is a vector(because minFunc expects the parameters to be a vector).
// We first convert theta to the(W1, W2, b1, b2) matrix / vector format, so that this
// follows the notation convention of the lecture notes.
// return: [cost, grad]
std::pair<float, cv::Mat> sparseAutoencoderCost(cv::Mat theta, int visibleSize, int hiddenSize, float lambda, float sparsityParam, float beta, cv::Mat data)
{
	// unroll the parameter-vector:
	int vs = visibleSize;
	int hs = hiddenSize;
	// We have to do the reshape of W1 and W2 the other way round than in Matlab, e.g. use the other dimension and then transpose.
	// Because Matlab reshapes by filling the columns. First column gets filled, then second, ... OpenCV fills up the rows of the new matrix first.
	Mat W1 = theta.rowRange(0, hs * vs).reshape(0, vs).t();
	Mat W2 = theta.rowRange(hs * vs, hs * vs * 2).reshape(0, hs).t();
	Mat b1 = theta.rowRange(hs * vs * 2, hs * vs * 2 + hs);
	Mat b2 = theta.rowRange(hs * vs * 2 + hs, hs * vs * 2 + hs + vs);

	float Jcost = 0.0f;
	float Jweight = 0.0f;
	float Jsparse = 0.0f;

	auto dataDim = data.rows;
	auto numExamples = data.cols;

	// Feedforward pass : For each example individually. Matrix notation faster.
	// For each of the 10000 values here we get a activation for each layer.
	// lec notes page 5, top
	// and page 9, step 1.
	Mat z2 = W1 * data + cv::repeat(b1, 1, numExamples); // z^(2) = W^(1)x + b^(1). Calculate activation of layer 2, given layer 1 and W1, b1.
	Mat a2 = sigmoid(z2); // a2 = final output activation of L2
	Mat z3 = W2 * a2 + cv::repeat(b2, 1, numExamples); // same for L3
	Mat a3 = sigmoid(z3);

	// Jcost = (0.5 / numExamples)*sum(sum((a3 - data). ^ 2)); % page 6, formula(8) lower
	Jcost = (0.5f / numExamples) * cv::norm(a3, data, cv::NORM_L2SQR); // page 6, formula (8) lower

	Jweight = 0.5f * (cv::norm(W1, cv::NORM_L2SQR) + cv::norm(W2, cv::NORM_L2SQR)); // regularisation (weight decay term)

	// average activation of each hidden unit(averaged over the training set).Page 14
	Mat a2RowSum;
	cv::reduce(a2, a2RowSum, 1, cv::REDUCE_SUM);
	Mat rho = (1.0f / numExamples) * a2RowSum;
	// we want rho to be close to sparsityParam, and we'll punish deviations (using the KL div.)
	// Jsparse = sum(sparsityParam.*log(sparsityParam. / rho) + (1 - sparsityParam).*log((1 - sparsityParam). / (1 - rho))); // sum over all neurons(except bias) of L2
	Mat logSoverRho, logOneMinusSoverOneMinusRho;
	cv::log(sparsityParam / rho, logSoverRho);
	cv::log((1 - sparsityParam) / (1 - rho), logOneMinusSoverOneMinusRho);
	Jsparse = cv::sum(sparsityParam * logSoverRho + (1 - sparsityParam) * logOneMinusSoverOneMinusRho)[0];
	
	float cost = Jcost + lambda * Jweight + beta * Jsparse;

	// Step 2 page 9
	// d3 = -(data - a3).*sigmoidInv(z3); % delta's of output layer, directly calculated using the labels (or in the AutoEnc (unsup. learning) case, the original data, that's what we want to reconstruct
	// we don't need sigmoidInv, use the expression mentioned in page9:
	// d3 = -(data-a3).* (a3.*(1-a3)); % the second whole part in () is f'(z_i^l)
	Mat d3 = -(data - a3).mul(a3.mul(1 - a3));
	
	Mat sterm = beta * (-sparsityParam / rho + (1 - sparsityParam) / (1 - rho));
	// Step 3 page 9
	// % d2 = (W2'*d3+repmat(sterm,1,numExamples)).*sigmoidInv(z2); 
	// d2 = (W2'*d3+repmat(sterm,1,numExamples)).* (a2.*(1-a2)); 
	// % d2 = (W2'*d3).* (a2.*(1-a2)); % same 'trick' for f'(z)
	Mat d2 = (W2.t() * d3 + cv::repeat(sterm, 1, numExamples)).mul(a2.mul(1 - a2));

	// Step 4 page 9
	// W1grad
	Mat W1grad = d2 * data.t();
	W1grad = (1.0f / numExamples) * W1grad + lambda * W1;

	// W2grad
	Mat W2grad = d3 * a2.t();
	W2grad = (1.0f / numExamples) * W2grad + lambda * W2;

	// b1grad
	Mat d2RowSum;
	cv::reduce(d2, d2RowSum, 1, cv::REDUCE_SUM);
	Mat b1grad = d2RowSum; // sum over each training example, see page 9 bottom 2c.
	b1grad = (1.0f / numExamples) * b1grad;

	// b2grad
	Mat d3RowSum;
	cv::reduce(d3, d3RowSum, 1, cv::REDUCE_SUM);
	Mat b2grad = d3RowSum;
	b2grad = (1.0f / numExamples) * b2grad;

	// convert gradients to vector format for the optimiser
	Mat grad;
	cv::vconcat(W1grad.reshape(1, hiddenSize * visibleSize), W2grad.reshape(1, hiddenSize * visibleSize), grad);
	cv::vconcat(grad, b1grad, grad);
	cv::vconcat(grad, b2grad, grad);
	return std::make_pair(cost, grad);
}

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				  "specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: autoEncoderPlayground [options]\n";
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);

	}
	catch (po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	LogLevel logLevel;
	if(boost::iequals(verboseLevelConsole, "PANIC")) logLevel = LogLevel::Panic;
	else if(boost::iequals(verboseLevelConsole, "ERROR")) logLevel = LogLevel::Error;
	else if(boost::iequals(verboseLevelConsole, "WARN")) logLevel = LogLevel::Warn;
	else if(boost::iequals(verboseLevelConsole, "INFO")) logLevel = LogLevel::Info;
	else if(boost::iequals(verboseLevelConsole, "DEBUG")) logLevel = LogLevel::Debug;
	else if(boost::iequals(verboseLevelConsole, "TRACE")) logLevel = LogLevel::Trace;
	else {
		cout << "Error: Invalid LogLevel." << endl;
		return EXIT_FAILURE;
	}
	
	Loggers->getLogger("imageio").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("SAE").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("SAE");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));
	
	path imagesroot(R"(C:\Users\Patrik\Documents\GitHub\FeatureDetection\autoEncoderPlayground\share\data\)");

	int visibleSize = 8*8; // 2 * 2;   // number of input units
	int hiddenSize = 25; // 2;     // number of hidden units
	float sparsityParam = 0.01f;   // desired average activation of the hidden units. rho
	float lambda = 0.0001f;     // weight decay parameter
	float beta = 3.0f;            // weight of sparsity penalty term

	vector<Mat> images;
	for (auto&& i = 1; i <= 10; ++i) {
		string imagepath = imagesroot.string() + string("image_") + std::to_string(i) + string(".png");
		images.emplace_back(cv::imread(imagepath, cv::IMREAD_GRAYSCALE));
	}

	// randomly sample patches from the images, i.e. prepare the input to the AE training
	vector<Mat> features = generatePatches(images, 8, 10000); // input. 8x8 patches, 10000.
	
	// Testing:
/*	Mat asdf = features[0];
	features.clear();
	Mat test(4, 1, CV_32FC1);
	test.at<float>(0, 0) = 0.3487f; // -0.1;
	test.at<float>(1, 0) = 0.6715f; // 0.3;
	test.at<float>(2, 0) = 0.5101f; // 0.1;
	test.at<float>(3, 0) = 0.4697f; // 0.05;
	features.push_back(test);*/
	// End testing

	features = normalizeData(features); // well, some strange normalization to [0.1, 0.9], mean/sdev normalization etc... tiny-cnn seems to do the 0.1, 0.9 thing too? (even 0.8?)

	Mat theta; ///< These are all our weights (including bias-terms) concatenated to a column-vector. 
	theta = initializeParameters(hiddenSize, visibleSize);
	// Note: I think we should make a class Parameter (CostParameter?...) that manages the en/un-rolling.
	// If we keep it like this, we need to keep track of hiddenSize etc.

	// we need the data in matrix form anyway, so here we go:
	Mat data(features[0].rows * features[0].cols, static_cast<int>(features.size()), CV_32FC1); // each column is 1 patch
	for (auto i = 0; i < features.size(); ++i) {
		Mat reshaped = features[i].reshape(1, features[0].rows * features[0].cols);
		reshaped.copyTo(data.col(i));
	}

	float cost;
	Mat grad;
	// Test:
	//Mat theta_t = (cv::Mat_<float>(22, 1) << -0.8847, -0.5834, 0.0299, 0.6047, -0.0473, 0.0309, -0.6009, 0.4323, -0.5413, 0.3190, -0.4729, 0.6570, 0.6836, 0.0218, -0.7288, -0.3234, 0, 0, 0, 0, 0, 0);
	 

//	std::tie(cost, grad) = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data);
	// Todo: Test this with a smaller model. I.e. 10 training examples, 3 hidden units.
	//auto test = std::bind(sparseAutoencoderCost, std::placeholders::_1, visibleSize, hiddenSize, lambda, sparsityParam, beta, data);
	//computeNumericalGradient(test, theta);
	// Actually we don't need to compute the gradients numerically. We use the ones obtained from the backprop. It's only to test the backprop gradients.
	// But our bprop gradients should be ~identical to the numerical gradients. Maybe they're off because of float vs double?
	
	// Randomly initialise the parameters:
//	theta = initializeParameters(hiddenSize, visibleSize);
	// we need a LBFGS, CG, SGD or LM optimiser now...
	// Just use a simple, stupid gradient descent for now: (tiny-cnn use "only" a simple SGD too)
/*	for (int i = 0; i < 5000; ++i) {
		std::tie(cost, grad) = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data);
		theta -= 0.35f * grad; // tiny-cnn use 0.003 ("learning rate")
		appLogger.info("Iter: " + std::to_string(i) + "; Cost: " + std::to_string(cost) + "; ||g||^2: " + std::to_string(cv::norm(grad, cv::NORM_L2)));
	}*/

	instance netParam;
	netParam.visibleSize = 8 * 8;
	netParam.hiddenSize = 25;
	netParam.sparsityParam = 0.01;
	netParam.lambda = 0.0001;
	netParam.beta = 3.0;
	Mat data64; data.convertTo(data64, CV_64FC1);
	netParam.data = data64;
	//Initialize the parameters for lbfgs
	lbfgsfloatval_t cost_l;
	lbfgsfloatval_t* theta_l = lbfgs_malloc(theta.rows);
	lbfgs_parameter_t optParam;
	if (theta_l == NULL) {
		cout << "ERROR: Failed to allocate a memory block for parameters." << endl;
		return 1;
	}
	theta = initializeParameters(hiddenSize, visibleSize);
	theta.convertTo(theta, CV_64FC1);
	memcpy(theta_l, theta.ptr<double>(0), theta.rows * sizeof(double)); //theta_l = theta.ptr<double>(0); doesn't work because lbfgs_malloc allocates MORE than 'n' memory!!!
	lbfgs_parameter_init(&optParam);
	optParam.max_iterations = 400;

	int ret = lbfgs(theta.rows, theta_l, &cost_l, sparseAECost, progress, (void*)(&netParam), &optParam);

	// cv::DownhillSolver and cv::ConjGradSolver, both didn't work
	/*Mat data64;*/ data.convertTo(data64, CV_64FC1);
	cv::Ptr<cv::MinProblemSolver::Function> ptr_F(new AECost(visibleSize, hiddenSize, lambda, sparsityParam, beta, data64));
/*	Mat simplexInitialStep(theta.rows, 1, CV_64FC1);
	
	float r = sqrt(6.0f) / static_cast<float>(sqrt(hiddenSize + visibleSize + 1));   // we'll choose weights uniformly from the interval [-r, r]
	std::random_device rd; ///< Used for seeding the PRNG (mt19937)
	std::mt19937 engine(rd()); ///< A Mersenne twister mt19937 engine
	std::uniform_real_distribution<double> randomNumber(-r, r);
	for (int r = 0; r < simplexInitialStep.rows; ++r) {
		simplexInitialStep.at<double>(r) = randomNumber(engine);
	}
*/
	//cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create(ptr_F, simplexInitialStep);
	cv::Ptr<cv::ConjGradSolver> solver = cv::ConjGradSolver::create(ptr_F);
	theta.convertTo(theta, CV_64FC1);
	auto res = solver->minimize(theta);

	// same as in cost-func, reshape "the other way", col-wise
	Mat W1img = theta.rowRange(0, hiddenSize * visibleSize).reshape(0, visibleSize).t();
	W1img = W1img.t();
	auto a = cv::mean(W1img)[0];
	W1img = W1img - cv::mean(W1img)[0];
	int L = W1img.rows; // 64
	int M = W1img.cols; // 25
	int size = std::sqrt(L);
	int n = std::sqrt(M); // breaks for dimensions that don't have a integer square root. see display_network.m code
	int m = n;
	// we got 5 * 5 = 25 weights to visualise
	// each weight is a 8 x 8 patch

	Mat array = -Mat::ones(1 + m * (size + 1), 1 + n * (size + 1), CV_32FC1);

	int k = 0;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			// every col of A is an image
			Mat tmp = cv::abs(W1img.col(k));
			double minVal, maxVal;
			cv::minMaxLoc(tmp, &minVal, &maxVal);
			float clim = maxVal;
			//Mat currImg = W1img.col(k).clone().reshape(0, 8);
			Mat arrayRangeThisImg = array.colRange(1 + i*(size + 1), 1 + i*(size + 1) + size).rowRange(1 + j*(size + 1), 1 + j*(size + 1) + size);
			Mat asdf = W1img.col(k).clone().reshape(1, size).t(); // We also need to transpose here I think, take the data row-wise
			asdf = asdf / clim;
			asdf.copyTo(arrayRangeThisImg);
			++k;
		}
	}
	cv::namedWindow("SAE");
	cv::resize(array, array, cv::Size(250, 250));
	cv::imshow("SAE", array);
	cv::waitKey(0);
	return 0;
}
