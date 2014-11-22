#include "gtest/gtest.h"

#include "superviseddescent/superviseddescent.hpp"

#include "opencv2/core/core.hpp"
/*#ifdef WIN32
#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif*/
#include "boost/math/special_functions/erf.hpp"

#include <vector>

using cv::Mat;
using std::vector;
namespace v2 = superviseddescent::v2;

template<typename ForwardIterator, typename T>
void strided_iota(ForwardIterator first, ForwardIterator last, T value, T stride)
{
	while (first != last) {
		*first++ = value;
		value += stride;
	}
}

TEST(SupervisedDescentOptimiser, SinConvergence) {
	// sin(x):
	auto h = [](Mat value) { return std::sin(value.at<float>(0)); };
	auto h_inv = [](float value) {
		if (value >= 1.0f) // our upper border of y is 1.0f, but it can be a bit larger due to floating point representation. asin then returns NaN.
			return std::asin(1.0f);
		else
			return std::asin(value);
		};

	float startInterval = -1.0f; float stepSize = 0.2f; int numValues = 11; Mat y_tr(numValues, 1, CV_32FC1); // sin: [-1:0.2:1]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	v2::SupervisedDescentOptimiser<v2::LinearRegressor> sdo({ v2::LinearRegressor() });
	
	// Test the callback mechanism as well: (better move to a separate unit test?)
	auto logNormalisedLSResidual = [&](const Mat& currentX){
		double differenceNorm = cv::norm(currentX, x_tr, cv::NORM_L2);
		double residual = differenceNorm / cv::norm(x_tr, cv::NORM_L2);
		EXPECT_DOUBLE_EQ(0.21369851877468238, residual);
	};

	sdo.train(x_tr, y_tr, x0, h, logNormalisedLSResidual);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	double trainingResidual = sdo.test(x_tr, y_tr, x0, h);
	EXPECT_DOUBLE_EQ(0.21369851877468238, trainingResidual);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -1.0f; float stepSizeTest = 0.05f; int numValuesTest = 41; Mat y_ts(numValuesTest, 1, CV_32FC1); // sin: [-1:0.05:1]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.
	
	double testResidual = sdo.test(x_ts_gt, y_ts, x0_ts, h); // x_ts_gt will only be used to calculate the residual
	ASSERT_DOUBLE_EQ(0.18001012289821938, testResidual);
}

TEST(SupervisedDescentOptimiser, SinConvergenceCascade) {
	// sin(x):
	auto h = [](Mat value) { return std::sin(value.at<float>(0)); };
	auto h_inv = [](float value) {
		if (value >= 1.0f) // our upper border of y is 1.0f, but it can be a bit larger due to floating point representation. asin then returns NaN.
			return std::asin(1.0f);
		else
			return std::asin(value);
	};

	float startInterval = -1.0f; float stepSize = 0.2f; int numValues = 11; Mat y_tr(numValues, 1, CV_32FC1); // sin: [-1:0.2:1]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	vector<v2::LinearRegressor> regressors;
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	v2::SupervisedDescentOptimiser<v2::LinearRegressor> sdo(regressors);
	sdo.train(x_tr, y_tr, x0, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	double trainingResidual = sdo.test(x_tr, y_tr, x0, h);
	EXPECT_DOUBLE_EQ(0.040279395431369915, trainingResidual);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -1.0f; float stepSizeTest = 0.05f; int numValuesTest = 41; Mat y_ts(numValuesTest, 1, CV_32FC1); // sin: [-1:0.05:1]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	double testResidual = sdo.test(x_ts_gt, y_ts, x0_ts, h); // x_ts_gt will only be used to calculate the residual
	ASSERT_DOUBLE_EQ(0.026156775112579144, testResidual);
}

TEST(SupervisedDescentOptimiser, XCubeConvergence) {
	// x^3:
	auto h = [](Mat value) { return std::pow(value.at<float>(0), 3); };
	auto h_inv = [](float value) { return std::cbrt(value); }; // cubic root

	float startInterval = -27.0f; float stepSize = 3.0f; int numValues = 19; Mat y_tr(numValues, 1, CV_32FC1); // cube: [-27:3:27]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	v2::SupervisedDescentOptimiser<v2::LinearRegressor> sdo({ v2::LinearRegressor() });
	sdo.train(x_tr, y_tr, x0, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	double trainingResidual = sdo.test(x_tr, y_tr, x0, h);
	EXPECT_DOUBLE_EQ(0.34416553222013058, trainingResidual);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -27.0f; float stepSizeTest = 0.5f; int numValuesTest = 109; Mat y_ts(numValuesTest, 1, CV_32FC1); // cube: [-27:0.5:27]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	double testResidual = sdo.test(x_ts_gt, y_ts, x0_ts, h); // x_ts_gt will only be used to calculate the residual
	ASSERT_DOUBLE_EQ(0.35342861514516261, testResidual);
}

TEST(SupervisedDescentOptimiser, XCubeConvergenceCascade) {
	// x^3:
	auto h = [](Mat value) { return std::pow(value.at<float>(0), 3); };
	auto h_inv = [](float value) { return std::cbrt(value); }; // cubic root

	float startInterval = -27.0f; float stepSize = 3.0f; int numValues = 19; Mat y_tr(numValues, 1, CV_32FC1); // cube: [-27:3:27]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	vector<v2::LinearRegressor> regressors;
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	v2::SupervisedDescentOptimiser<v2::LinearRegressor> sdo(regressors);
	sdo.train(x_tr, y_tr, x0, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	double trainingResidual = sdo.test(x_tr, y_tr, x0, h);
	EXPECT_DOUBLE_EQ(0.043127251144503408, trainingResidual);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -27.0f; float stepSizeTest = 0.5f; int numValuesTest = 109; Mat y_ts(numValuesTest, 1, CV_32FC1); // cube: [-27:0.5:27]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	double testResidual = sdo.test(x_ts_gt, y_ts, x0_ts, h); // x_ts_gt will only be used to calculate the residual
	ASSERT_DOUBLE_EQ(0.058898552312675233, testResidual);
}

TEST(SupervisedDescentOptimiser, ErfConvergence) {
	// erf(x):
	auto h = [](Mat value) { return std::erf(value.at<float>(0)); };
	auto h_inv = [](float value) { return boost::math::erf_inv(value); };

	float startInterval = -0.99f; float stepSize = 0.11f; int numValues = 19; Mat y_tr(numValues, 1, CV_32FC1); // erf: [-0.99:0.11:0.99]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	v2::SupervisedDescentOptimiser<v2::LinearRegressor> sdo({ v2::LinearRegressor() });
	sdo.train(x_tr, y_tr, x0, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	double trainingResidual = sdo.test(x_tr, y_tr, x0, h);
	EXPECT_DOUBLE_EQ(0.30944183062755731, trainingResidual);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -0.99f; float stepSizeTest = 0.03f; int numValuesTest = 67; Mat y_ts(numValuesTest, 1, CV_32FC1); // erf: [-0.99:0.03:0.99]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	double testResidual = sdo.test(x_ts_gt, y_ts, x0_ts, h); // x_ts_gt will only be used to calculate the residual
	ASSERT_DOUBLE_EQ(0.25736006623771107, testResidual);
}

TEST(SupervisedDescentOptimiser, ErfConvergenceCascade) {
	// erf(x):
	auto h = [](Mat value) { return std::erf(value.at<float>(0)); };
	auto h_inv = [](float value) { return boost::math::erf_inv(value); };

	float startInterval = -0.99f; float stepSize = 0.11f; int numValues = 19; Mat y_tr(numValues, 1, CV_32FC1); // erf: [-0.99:0.11:0.99]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	vector<v2::LinearRegressor> regressors;
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	v2::SupervisedDescentOptimiser<v2::LinearRegressor> sdo(regressors);
	sdo.train(x_tr, y_tr, x0, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	double trainingResidual = sdo.test(x_tr, y_tr, x0, h);
	EXPECT_DOUBLE_EQ(0.069510675228435778, trainingResidual);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = -0.99f; float stepSizeTest = 0.03f; int numValuesTest = 67; Mat y_ts(numValuesTest, 1, CV_32FC1); // erf: [-0.99:0.03:0.99]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	double testResidual = sdo.test(x_ts_gt, y_ts, x0_ts, h); // x_ts_gt will only be used to calculate the residual
	ASSERT_DOUBLE_EQ(0.046327173469310361, testResidual);
}

TEST(SupervisedDescentOptimiser, ExpConvergence) {
	// exp(x):
	auto h = [](Mat value) { return std::exp(value.at<float>(0)); };
	auto h_inv = [](float value) { return std::log(value); };

	float startInterval = 1.0f; float stepSize = 3.0f; int numValues = 10; Mat y_tr(numValues, 1, CV_32FC1); // exp: [1:3:28]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	v2::SupervisedDescentOptimiser<v2::LinearRegressor> sdo({ v2::LinearRegressor() });
	sdo.train(x_tr, y_tr, x0, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	double trainingResidual = sdo.test(x_tr, y_tr, x0, h);
	EXPECT_DOUBLE_EQ(0.19952251597692217, trainingResidual);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = 1.0f; float stepSizeTest = 0.5f; int numValuesTest = 55; Mat y_ts(numValuesTest, 1, CV_32FC1); // exp: [1:0.5:28]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	double testResidual = sdo.test(x_ts_gt, y_ts, x0_ts, h); // x_ts_gt will only be used to calculate the residual
	ASSERT_DOUBLE_EQ(0.19245695013636951, testResidual);
}

TEST(SupervisedDescentOptimiser, ExpConvergenceCascade) {
	// exp(x):
	auto h = [](Mat value) { return std::exp(value.at<float>(0)); };
	auto h_inv = [](float value) { return std::log(value); };

	float startInterval = 1.0f; float stepSize = 3.0f; int numValues = 10; Mat y_tr(numValues, 1, CV_32FC1); // exp: [1:3:28]
	{
		vector<float> values(numValues);
		strided_iota(std::begin(values), std::next(std::begin(values), numValues), startInterval, stepSize);
		y_tr = Mat(values, true);
	}
	Mat x_tr(numValues, 1, CV_32FC1); // Will be the inverse of y_tr
	{
		vector<float> values(numValues);
		std::transform(y_tr.begin<float>(), y_tr.end<float>(), begin(values), h_inv);
		x_tr = Mat(values, true);
	}

	Mat x0 = 0.5f * Mat::ones(numValues, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	vector<v2::LinearRegressor> regressors;
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	regressors.emplace_back(v2::LinearRegressor());
	v2::SupervisedDescentOptimiser<v2::LinearRegressor> sdo(regressors);
	sdo.train(x_tr, y_tr, x0, h);

	// Make sure the training converges, i.e. the residual is correct on the training data:
	double trainingResidual = sdo.test(x_tr, y_tr, x0, h);
	EXPECT_DOUBLE_EQ(0.025108688779870918, trainingResidual);

	// Test the trained model:
	// Test data with finer resolution:
	float startIntervalTest = 1.0f; float stepSizeTest = 0.5f; int numValuesTest = 55; Mat y_ts(numValuesTest, 1, CV_32FC1); // exp: [1:0.5:28]
	{
		vector<float> values(numValuesTest);
		strided_iota(std::begin(values), std::next(std::begin(values), numValuesTest), startIntervalTest, stepSizeTest);
		y_ts = Mat(values, true);
	}
	Mat x_ts_gt(numValuesTest, 1, CV_32FC1); // Will be the inverse of y_ts
	{
		vector<float> values(numValuesTest);
		std::transform(y_ts.begin<float>(), y_ts.end<float>(), begin(values), h_inv);
		x_ts_gt = Mat(values, true);
	}
	Mat x0_ts = 0.5f * Mat::ones(numValuesTest, 1, CV_32FC1); // fixed initialization x0 = c = 0.5.

	double testResidual = sdo.test(x_ts_gt, y_ts, x0_ts, h); // x_ts_gt will only be used to calculate the residual
	ASSERT_DOUBLE_EQ(0.012534944044555876, testResidual);
}
