/*
 * TrackingBenchmark.hpp
 *
 *  Created on: 17.02.2014
 *      Author: poschmann
 */

#ifndef TRACKINGBENCHMARK_HPP_
#define TRACKINGBENCHMARK_HPP_

#include "logging/Logger.hpp"
#include "imageio/AnnotatedImageSource.hpp"
#include "imageio/AnnotationSink.hpp"
#include "classification/ExampleManagement.hpp"
#include "classification/BinaryClassifier.hpp"
#include "classification/TrainableProbabilisticSvmClassifier.hpp"
#include "condensation/CondensationTracker.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/filesystem.hpp"
#include "boost/property_tree/ptree.hpp"
#include <memory>
#include <string>

using namespace imageio;
using namespace imageprocessing;
using namespace condensation;
using namespace classification;
using cv::Mat;
using boost::property_tree::ptree;
using boost::filesystem::path;
using boost::optional;
using std::pair;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

/**
 * Benchmark for tracking algorithms.
 */
class TrackingBenchmark {
public:

	/**
	 * Constructs a new tracking benchmark, constructing the tracker from the given configuration.
	 *
	 * @param[in] config The configuration of the tracker.
	 */
	TrackingBenchmark(ptree& config);

	/**
	 * Tests the tracker using the test sequences given by the config.
	 *
	 * @param[in] resultsDirectory The directory to write the detailled results into (directory for each test).
	 * @param[in] count The number of runs per test.
	 * @param[in] config The configuration of the tests.
	 * @param[in] log The logger for reporting overall results.
	 * @return The score of the tracker over all tests in percent.
	 */
	double runTests(const path& resultsDirectory, size_t count, const ptree& config, logging::Logger& log);

	/**
	 * Runs a single test on the tracker.
	 *
	 * @param[in] imageSource The source of the annotated images from the test sequence.
	 * @param[in] annotationSink The sink for writing the tracking result (bounding boxes) into.
	 * @return A pair containing the FPS values (overall and tracking only (without loading images etc)).
	 */
	pair<double, double> runTest(shared_ptr<AnnotatedImageSource> imageSource, shared_ptr<AnnotationSink> annotationSink);

	/**
	 * Computes the overlap (ratio of intersection to union) of two bounding boxes.
	 *
	 * @param[in] a The first box.
	 * @param[in] b The second box.
	 * @return The overlap.
	 */
	double computeOverlap(cv::Rect a, cv::Rect b) const;

private:

	unique_ptr<ExampleManagement> createExampleManagement(ptree& config, shared_ptr<BinaryClassifier> classifier, bool positive);
	shared_ptr<TrainableProbabilisticSvmClassifier> createSvm(ptree& config);
	void initTracking(ptree& config);

	unique_ptr<CondensationTracker> tracker; ///< The tracker under test.
};

#endif /* TRACKINGBENCHMARK_HPP_ */
