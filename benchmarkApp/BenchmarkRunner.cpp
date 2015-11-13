/*
 * BenchmarkRunner.cpp
 *
 *  Created on: 26.09.2013
 *      Author: poschmann
 */

#include "Benchmark.hpp"
#include "imageio/ImageSource.hpp"
#include "imageio/LandmarkSource.hpp"
#include "imageio/VideoImageSource.hpp"
#include "imageio/DirectoryImageSource.hpp"
#include "imageio/BobotLandmarkSource.hpp"
#include "imageio/OrderedLabeledImageSource.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/DirectPyramidFeatureExtractor.hpp"
#include "imageprocessing/DirectImageFeatureExtractor.hpp"
#include "imageprocessing/IntegralFeatureExtractor.hpp"
#include "imageprocessing/PatchResizingFeatureExtractor.hpp"
#include "imageprocessing/ConversionFilter.hpp"
#include "imageprocessing/ExtendedHogFilter.hpp"
#include "imageprocessing/GradientBinningFilter.hpp"
#include "imageprocessing/GradientFilter.hpp"
#include "imageprocessing/GradientSumFilter.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/HaarFeatureFilter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"
#include "imageprocessing/HogFilter.hpp"
#include "imageprocessing/LbpFilter.hpp"
#include "imageprocessing/IntegralGradientFilter.hpp"
#include "imageprocessing/IntegralImageFilter.hpp"
#include "imageprocessing/PyramidHogFilter.hpp"
#include "imageprocessing/ResizingFilter.hpp"
#include "imageprocessing/SpatialHistogramFilter.hpp"
#include "imageprocessing/SpatialPyramidHistogramFilter.hpp"
#include "imageprocessing/UnitNormFilter.hpp"
#include "imageprocessing/WhiteningFilter.hpp"
#include "classification/HistogramIntersectionKernel.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/RbfKernel.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/UnlimitedExampleManagement.hpp"
#include "classification/AgeBasedExampleManagement.hpp"
#include "classification/ConfidenceBasedExampleManagement.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/TrainableProbabilisticSvmClassifier.hpp"
#include "classification/FixedTrainableProbabilisticSvmClassifier.hpp"
#include "libsvm/LibSvmClassifier.hpp"
#ifdef WITH_LIBLINEAR_CLASSIFIER
	#include "liblinear/LibLinearClassifier.hpp"
#endif
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"
#include "boost/optional/optional.hpp"
#include "boost/filesystem.hpp"
#include <memory>
#include <iostream>
#include <sstream>

using namespace imageio;
using namespace imageprocessing;
using namespace classification;
using libsvm::LibSvmClassifier;
using cv::Size;
using boost::property_tree::ptree;
using boost::property_tree::info_parser::read_info;
using boost::optional;
using boost::filesystem::path;
using boost::filesystem::exists;
using boost::filesystem::is_regular_file;
using boost::filesystem::create_directory;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using std::istringstream;
using std::invalid_argument;

LbpFilter::Type getLbpType(string type) {
	if (type == "lbp8")
		return LbpFilter::Type::LBP8;
	else if (type == "lbp8uniform")
		return LbpFilter::Type::LBP8_UNIFORM;
	else if (type == "lbp4")
		return LbpFilter::Type::LBP4;
	else if (type == "lbp4rotated")
		return LbpFilter::Type::LBP4_ROTATED;
	else
		throw invalid_argument("invalid LBP type: " + type);
}

HistogramFilter::Normalization getNormalizationType(string type) {
	if (type == "none")
		return HistogramFilter::Normalization::NONE;
	else if (type == "l2norm")
		return HistogramFilter::Normalization::L2NORM;
	else if (type == "l2hys")
		return HistogramFilter::Normalization::L2HYS;
	else if (type == "l1norm")
		return HistogramFilter::Normalization::L1NORM;
	else if (type == "l1sqrt")
		return HistogramFilter::Normalization::L1SQRT;
	else
		throw invalid_argument("invalid normalization type: " + type);
}

shared_ptr<HistogramFilter> createHistogramFilter(unsigned int bins, ptree& config) {
	if (config.get_value<string>() == "spatial")
		return make_shared<SpatialHistogramFilter>(bins, config.get<int>("cellSize"), config.get<int>("blockSize"),
				config.get<bool>("interpolate"), config.get<bool>("concatenate"), getNormalizationType(config.get<string>("normalization")));
	else 	if (config.get_value<string>() == "pyramid")
		return make_shared<SpatialPyramidHistogramFilter>(bins, config.get<int>("levels"),
				config.get<bool>("interpolate"), getNormalizationType(config.get<string>("normalization")));
	else
		throw invalid_argument("invalid histogram type: " + config.get_value<string>());
}

shared_ptr<ImageFilter> createHogFilter(int bins, ptree& config) {
	if (config.get_value<string>() == "spatial") {
		if (config.get<int>("blockSize") == 1 && !config.get<bool>("signedAndUnsigned"))
			return createHistogramFilter(bins, config);
		else
			return make_shared<HogFilter>(bins, config.get<int>("cellSize"), config.get<int>("blockSize"),
					config.get<bool>("interpolate"), config.get<bool>("signedAndUnsigned"));
	} else if (config.get_value<string>() == "pyramid") {
		return make_shared<PyramidHogFilter>(bins, config.get<int>("levels"), config.get<bool>("interpolate"), config.get<bool>("signedAndUnsigned"));
	} else {
		throw invalid_argument("invalid histogram type: " + config.get_value<string>());
	}
}

shared_ptr<DirectPyramidFeatureExtractor> createPyramidExtractor(ptree& config, float sizeScale) {
	float scaleFactor = 1.f / sizeScale;
	if (config.get<string>("scaleFactor") != "auto")
		scaleFactor = config.get<float>("scaleFactor");
	shared_ptr<DirectPyramidFeatureExtractor> pyramidExtractor = make_shared<DirectPyramidFeatureExtractor>(
			config.get<int>("patch.width"), config.get<int>("patch.height"),
			config.get<int>("patch.minWidth"), config.get<int>("patch.maxWidth"), scaleFactor);
	pyramidExtractor->addImageFilter(make_shared<GrayscaleFilter>());
	return pyramidExtractor;
}

shared_ptr<DirectImageFeatureExtractor> createImageExtractor(ptree& config) {
	shared_ptr<DirectImageFeatureExtractor> extractor = make_shared<DirectImageFeatureExtractor>();
	extractor->addImageFilter(make_shared<GrayscaleFilter>());
	extractor->addPatchFilter(make_shared<ResizingFilter>(Size(config.get<int>("patch.width"), config.get<int>("patch.height"))));
	return extractor;
}

shared_ptr<FeatureExtractor> createExtractor(ptree& config, float sizeScale, vector<shared_ptr<ImageFilter>> filters) {
	if (config.get_value<string>() == "image") {
		shared_ptr<DirectImageFeatureExtractor> extractor = createImageExtractor(config);
		for (shared_ptr<ImageFilter>& filter : filters)
			extractor->addPatchFilter(filter);
		return extractor;
	} else if (config.get_value<string>() == "pyramid") {
		shared_ptr<DirectPyramidFeatureExtractor> extractor = createPyramidExtractor(config, sizeScale);
		for (shared_ptr<ImageFilter>& filter : filters)
			extractor->addPatchFilter(filter);
		return extractor;
	} else {
		throw invalid_argument("invalid base extractor type: " + config.get_value<string>());
	}
}

shared_ptr<FeatureExtractor> createGrayscaleExtractor(ptree& config, float sizeScale) {
	return createExtractor(config.get_child("base"), sizeScale, vector<shared_ptr<ImageFilter>>()); // {} crashes the VS2013 compiler
}

shared_ptr<FeatureExtractor> createHistEqExtractor(ptree& config, float sizeScale) {
	return createExtractor(config.get_child("base"), sizeScale, {
			make_shared<HistogramEqualizationFilter>()
	});
}

shared_ptr<FeatureExtractor> createWhiExtractor(ptree& config, float sizeScale) {
	return createExtractor(config.get_child("base"), sizeScale, {
			make_shared<WhiteningFilter>(),
			make_shared<HistogramEqualizationFilter>(),
			make_shared<ConversionFilter>(CV_32F, 1.0 / 127.5, -1.0),
			make_shared<UnitNormFilter>(cv::NORM_L2)
	});
}

shared_ptr<FeatureExtractor> createLbpExtractor(ptree& config, float sizeScale) {
	shared_ptr<LbpFilter> lbpFilter = make_shared<LbpFilter>(getLbpType(config.get<string>("type")));
	if (config.get<string>("base") == "pyramid") {
		shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = createPyramidExtractor(config.get_child("base"), sizeScale);
		featureExtractor->addLayerFilter(lbpFilter);
		featureExtractor->addPatchFilter(createHistogramFilter(lbpFilter->getBinCount(), config.get_child("histogram")));
		return featureExtractor;
	} else if (config.get<string>("base") == "image") {
		shared_ptr<DirectImageFeatureExtractor> featureExtractor = createImageExtractor(config.get_child("base"));
		featureExtractor->addPatchFilter(lbpFilter);
		featureExtractor->addPatchFilter(createHistogramFilter(lbpFilter->getBinCount(), config.get_child("histogram")));
		return featureExtractor;
	} else {
		throw invalid_argument("invalid base extractor type: " + config.get_value<string>());
	}
}

shared_ptr<FeatureExtractor> createHaarExtractor(ptree& config, float sizeScale) {
	vector<float> sizes;
	float size;
	istringstream sizesStream(config.get<string>("sizes"));
	while (sizesStream.good() && !sizesStream.fail()) {
		sizesStream >> size;
		sizes.push_back(size);
	}
	float gridRows = config.get<float>("gridRows");
	float gridCols = config.get<float>("gridCols");
	int types = 0;
	string type;
	istringstream typesStream(config.get<string>("types"));
	while (typesStream.good() && !typesStream.fail()) {
		typesStream >> type;
		if (type == "2rect")
			types |= HaarFeatureFilter::TYPE_2RECTANGLE;
		else if (type == "3rect")
			types |= HaarFeatureFilter::TYPE_3RECTANGLE;
		else if (type == "4rect")
			types |= HaarFeatureFilter::TYPE_4RECTANGLE;
		else if (type == "center-surround")
			types |= HaarFeatureFilter::TYPE_CENTER_SURROUND;
		else if (type == "all")
			types |= HaarFeatureFilter::TYPES_ALL;
	}
	shared_ptr<DirectImageFeatureExtractor> featureExtractor = make_shared<DirectImageFeatureExtractor>();
	featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
	featureExtractor->addImageFilter(make_shared<IntegralImageFilter>());
	featureExtractor->addPatchFilter(make_shared<HaarFeatureFilter>(sizes, gridRows, gridCols, types));
	return featureExtractor;
}

shared_ptr<FeatureExtractor> createHogExtractor(ptree& config, float sizeScale, shared_ptr<ImageFilter> hogFilter) {
	if (config.get<string>("gradients") == "patch") {
		if (config.get<string>("base") == "pyramid") {
			shared_ptr<DirectPyramidFeatureExtractor> featureExtractor = createPyramidExtractor(config.get_child("gradients.base"), sizeScale);
			featureExtractor->addLayerFilter(make_shared<GradientFilter>(config.get<int>("gradients.gradientKernel"), config.get<int>("gradients.blurKernel")));
			featureExtractor->addLayerFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed")));
			featureExtractor->addPatchFilter(hogFilter);
			return featureExtractor;
		} else if (config.get<string>("base") == "image") {
			shared_ptr<DirectImageFeatureExtractor> featureExtractor = createImageExtractor(config.get_child("gradients.base"));
			featureExtractor->addPatchFilter(make_shared<GradientFilter>(config.get<int>("gradients.gradientKernel"), config.get<int>("gradients.blurKernel")));
			featureExtractor->addPatchFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed")));
			featureExtractor->addPatchFilter(hogFilter);
			return featureExtractor;
		} else {
			throw invalid_argument("invalid base extractor type: " + config.get_value<string>());
		}
	} else if (config.get<string>("gradients") == "integral") {
		shared_ptr<DirectImageFeatureExtractor> featureExtractor = make_shared<DirectImageFeatureExtractor>();
		featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
		featureExtractor->addImageFilter(make_shared<IntegralImageFilter>());
		featureExtractor->addPatchFilter(make_shared<IntegralGradientFilter>(config.get<int>("gradients.count")));
		featureExtractor->addPatchFilter(make_shared<GradientBinningFilter>(config.get<int>("bins"), config.get<bool>("signed")));
		featureExtractor->addPatchFilter(hogFilter);
		return featureExtractor;
	} else {
		throw invalid_argument("invalid gradients type: " + config.get<string>("gradients"));
	}
}

shared_ptr<FeatureExtractor> createHogExtractor(ptree& config, float sizeScale) {
	return createHogExtractor(config, sizeScale, createHogFilter(config.get<int>("bins"), config.get_child("histogram")));
}

shared_ptr<FeatureExtractor> createEHogExtractor(ptree& config, float sizeScale) {
	return createHogExtractor(config, sizeScale, make_shared<ExtendedHogFilter>(
					config.get<int>("bins"),
					config.get<int>("histogram.cellSize"),
					config.get<bool>("histogram.interpolate"),
					config.get<bool>("histogram.signedAndUnsigned"),
					config.get<float>("histogram.alpha")));
}

shared_ptr<FeatureExtractor> createSurfExtractor(ptree& config, float sizeScale) {
	shared_ptr<DirectImageFeatureExtractor> featureExtractor = make_shared<DirectImageFeatureExtractor>();
	featureExtractor->addImageFilter(make_shared<GrayscaleFilter>());
	featureExtractor->addImageFilter(make_shared<IntegralImageFilter>());
	featureExtractor->addPatchFilter(make_shared<IntegralGradientFilter>(config.get<int>("gradientCount")));
	featureExtractor->addPatchFilter(make_shared<GradientSumFilter>(config.get<int>("cellCount")));
	featureExtractor->addPatchFilter(make_shared<UnitNormFilter>(cv::NORM_L2));
	return featureExtractor;
}

shared_ptr<FeatureExtractor> createFeatureExtractor(ptree& config, float sizeScale) {
	shared_ptr<FeatureExtractor> featureExtractor;
	if (config.get_value<string>() == "grayscale") {
		featureExtractor = createGrayscaleExtractor(config, sizeScale);
	} else if (config.get_value<string>() == "histeq") {
		featureExtractor = createHistEqExtractor(config, sizeScale);
	} else if (config.get_value<string>() == "whi") {
		featureExtractor = createWhiExtractor(config, sizeScale);
	} else if (config.get_value<string>() == "lbp") {
		featureExtractor = createLbpExtractor(config, sizeScale);
	} else if (config.get_value<string>() == "haar") {
		featureExtractor = createHaarExtractor(config, sizeScale);
	} else if (config.get_value<string>() == "hog") {
		featureExtractor = createHogExtractor(config, sizeScale);
	} else if (config.get_value<string>() == "ehog") {
		featureExtractor = createEHogExtractor(config, sizeScale);
	} else if (config.get_value<string>() == "surf") {
		featureExtractor = createSurfExtractor(config, sizeScale);
	} else {
		throw invalid_argument("invalid feature type: " + config.get_value<string>());
	}
	optional<float> scaleFactor = config.get_optional<float>("scale");
	if (scaleFactor)
		return make_shared<PatchResizingFeatureExtractor>(featureExtractor, *scaleFactor);
	return featureExtractor;
}

shared_ptr<Kernel> createKernel(ptree& config) {
	if (config.get_value<string>() == "rbf") {
		return make_shared<RbfKernel>(config.get<double>("gamma"));
	} else if (config.get_value<string>() == "poly") {
		return make_shared<PolynomialKernel>(
				config.get<double>("alpha"), config.get<double>("constant"), config.get<double>("degree"));
	} else if (config.get_value<string>() == "hik") {
		return make_shared<HistogramIntersectionKernel>();
	} else if (config.get_value<string>() == "linear") {
		return make_shared<LinearKernel>();
	} else {
		throw invalid_argument("invalid kernel type: " + config.get_value<string>());
	}
}

unique_ptr<ExampleManagement> createExampleManagement(ptree& config, shared_ptr<BinaryClassifier> classifier, bool positive) {
	if (config.get_value<string>() == "unlimited") {
		return unique_ptr<ExampleManagement>(new UnlimitedExampleManagement(config.get<size_t>("required")));
	} else if (config.get_value<string>() == "agebased") {
		return unique_ptr<ExampleManagement>(new AgeBasedExampleManagement(config.get<size_t>("capacity"), config.get<size_t>("required")));
	} else if (config.get_value<string>() == "confidencebased") {
		return unique_ptr<ExampleManagement>(new ConfidenceBasedExampleManagement(classifier, positive, config.get<size_t>("capacity"), config.get<size_t>("required")));
	} else {
		throw invalid_argument("invalid example management type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableSvmClassifier> createLibSvmClassifier(ptree& config, shared_ptr<Kernel> kernel) {
	if (config.get_value<string>() == "binary") {
		shared_ptr<LibSvmClassifier> trainableSvm = LibSvmClassifier::createBinarySvm(kernel, config.get<double>("C"));
		trainableSvm->setPositiveExampleManagement(
				unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("positiveExamples"), trainableSvm, true)));
		trainableSvm->setNegativeExampleManagement(
				unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("negativeExamples"), trainableSvm, false)));
		optional<ptree&> negativesConfig = config.get_child_optional("staticNegativeExamples");
		if (negativesConfig && negativesConfig->get_value<bool>()) {
			trainableSvm->loadStaticNegatives(negativesConfig->get<string>("filename"),
					negativesConfig->get<int>("amount"), negativesConfig->get<double>("scale"));
		}
		return trainableSvm;
	} else if (config.get_value<string>() == "one-class") {
		shared_ptr<LibSvmClassifier> trainableSvm = LibSvmClassifier::createOneClassSvm(kernel, config.get<double>("nu"));
		trainableSvm->setPositiveExampleManagement(
				unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("positiveExamples"), trainableSvm, true)));
		return trainableSvm;
	} else {
		throw invalid_argument("invalid libSVM training type: " + config.get_value<string>());
	}
}

shared_ptr<TrainableSvmClassifier> createLibLinearClassifier(ptree& config) {
#ifdef WITH_LIBLINEAR_CLASSIFIER
	shared_ptr<liblinear::LibLinearClassifier> trainableSvm = make_shared<liblinear::LibLinearClassifier>(config.get<double>("C"), config.get<bool>("bias"));
	trainableSvm->setPositiveExampleManagement(
			unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("positiveExamples"), trainableSvm, true)));
	trainableSvm->setNegativeExampleManagement(
			unique_ptr<ExampleManagement>(createExampleManagement(config.get_child("negativeExamples"), trainableSvm, false)));
	optional<ptree&> negativesConfig = config.get_child_optional("staticNegativeExamples");
	if (negativesConfig && negativesConfig->get_value<bool>()) {
		trainableSvm->loadStaticNegatives(negativesConfig->get<string>("filename"),
				negativesConfig->get<int>("amount"), negativesConfig->get<double>("scale"));
	}
	return trainableSvm;
#else
	throw std::runtime_error("Cannot load a LibLinear classifier. Run CMake with WITH_LIBLINEAR_CLASSIFIER set to ON to enable.");
#endif // WITH_LIBLINEAR_CLASSIFIER
}

shared_ptr<TrainableProbabilisticClassifier> createTrainableProbabilisticSvm(
		shared_ptr<TrainableSvmClassifier> trainableSvm, ptree& config) {
	shared_ptr<TrainableProbabilisticSvmClassifier> svm;
	if (config.get_value<string>() == "default")
		svm = make_shared<TrainableProbabilisticSvmClassifier>(trainableSvm,
				config.get<int>("positiveExamples"), config.get<int>("negativeExamples"),
				config.get<double>("positiveProbability"), config.get<double>("negativeProbability"));
	else if (config.get_value<string>() == "precomputed")
		svm = make_shared<FixedTrainableProbabilisticSvmClassifier>(trainableSvm,
				config.get<double>("positiveProbability"), config.get<double>("negativeProbability"),
				config.get<double>("positiveMean"), config.get<double>("negativeMean"));
	else if (config.get_value<string>() == "predefined")
		svm = make_shared<FixedTrainableProbabilisticSvmClassifier>(trainableSvm,
				config.get<double>("logisticA"), config.get<double>("logisticB"));
	else
		throw invalid_argument("invalid probabilistic SVM type: " + config.get_value<string>());
	if (config.get<string>("adjustThreshold") != "no")
		svm->setAdjustThreshold(config.get<double>("adjustThreshold"));
	return svm;
}

shared_ptr<TrainableProbabilisticClassifier> createClassifier(ptree& config) {
	if (config.get_value<string>() == "libSvm") {
		shared_ptr<Kernel> kernel = createKernel(config.get_child("kernel"));
		shared_ptr<TrainableSvmClassifier> trainableSvm = createLibSvmClassifier(config.get_child("training"), kernel);
		shared_ptr<SvmClassifier> svm = trainableSvm->getSvm();
		optional<ptree&> thresholdConfig = config.get_child_optional("threshold");
		if (thresholdConfig)
			svm->setThreshold(thresholdConfig->get_value<float>());
		return createTrainableProbabilisticSvm(trainableSvm, config.get_child("probabilistic"));
	} else if (config.get_value<string>() == "libLinear") {
		shared_ptr<TrainableSvmClassifier> trainableSvm = createLibLinearClassifier(config.get_child("training"));
		shared_ptr<SvmClassifier> svm = trainableSvm->getSvm();
		optional<ptree&> thresholdConfig = config.get_child_optional("threshold");
		if (thresholdConfig)
			svm->setThreshold(thresholdConfig->get_value<float>());
		return createTrainableProbabilisticSvm(trainableSvm, config.get_child("probabilistic"));
	} else {
		throw invalid_argument("invalid classifier type: " + config.get_value<string>());
	}
}

void addAlgorithm(Benchmark& benchmark, ptree& config, string directory) {
	ptree algorithm;
	algorithm.put_value(config.get_value<string>());
	algorithm.add_child("confidenceThreshold", config.get_child("confidenceThreshold"));
	algorithm.add_child("maxNegativesPerFrame", config.get_child("maxNegativesPerFrame"));
	algorithm.add_child("initialNegatives", config.get_child("initialNegatives"));
	algorithm.add_child("feature", config.get_child("feature"));
	algorithm.add_child("classifier", config.get_child("classifier"));
	ptree output;
	output.add_child("algorithm", algorithm);
	write_info(directory + '/' + config.get_value<string>(), output);
	benchmark.add(config.get_value<string>(),
			createFeatureExtractor(config.get_child("feature"), benchmark.getSizeScale()),
			createClassifier(config.get_child("classifier")),
			config.get<float>("confidenceThreshold"),
			config.get<size_t>("maxNegativesPerFrame"),
			config.get<size_t>("initialNegatives"));
}

void runTest(Benchmark& benchmark, ptree& config) {
	if (config.get<string>("type") != "bobot")
		throw invalid_argument("invalid test type: " + config.get<string>("type"));
	optional<string> file = config.get_optional<string>("file");
	optional<string> directory = config.get_optional<string>("directory");
	shared_ptr<ImageSource> imageSource;
	if (file && !directory)
		imageSource = make_shared<VideoImageSource>(*file);
	else if (!file && directory)
		imageSource = make_shared<DirectoryImageSource>(*directory);
	else
		throw invalid_argument("either a video file or a directory must be given for test " + config.get_value<string>());
	shared_ptr<LandmarkSource> landmarkSource = make_shared<BobotLandmarkSource>(config.get<string>("groundTruth"), imageSource);
	shared_ptr<LabeledImageSource> source = make_shared<OrderedLabeledImageSource>(imageSource, landmarkSource);
	benchmark.run(config.get_value<string>(), source);
}

int main(int argc, char **argv) {
	string configFile = "default.cfg";
	if (argc > 1)
		configFile = argv[1];
	ptree config;
	read_info(configFile, config);
	ptree& benchmarkConfig = config.get_child("benchmark");

	path directory(benchmarkConfig.get<string>("directory"));
	if (!exists(directory))
		create_directory(directory);
	else if (is_regular_file(directory))
		throw invalid_argument("a file named " + benchmarkConfig.get<string>("directory") + " prevents creating a directory with that name");
	Benchmark benchmark(benchmarkConfig.get<float>("sizeMin"), benchmarkConfig.get<float>("sizeMax"), benchmarkConfig.get<float>("sizeScale"),
			benchmarkConfig.get<float>("step"), benchmarkConfig.get<float>("allowedOverlap"), benchmarkConfig.get<string>("directory"));
	auto iterators = benchmarkConfig.equal_range("algorithm");
	for (auto it = iterators.first; it != iterators.second; ++it)
		addAlgorithm(benchmark, it->second, benchmarkConfig.get<string>("directory"));
	iterators = benchmarkConfig.equal_range("test");
	for (auto it = iterators.first; it != iterators.second; ++it)
		runTest(benchmark, it->second);
}
