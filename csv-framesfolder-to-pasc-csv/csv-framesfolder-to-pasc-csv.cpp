/*
 * csv-framesfolder-to-pasc-csv.cpp
 *
 *  Created on: 23.10.2014
 *      Author: Patrik Huber
 *
 * Example:
 * csv-framesfolder-to-pasc-csv ...
 *   
 */

#include <memory>
#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem.hpp"
#include "boost/archive/text_iarchive.hpp"

#include "facerecognition/pasc.hpp"
#include "facerecognition/utils.hpp"

#include "logging/LoggerFactory.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using logging::Logger;
using logging::LoggerFactory;
using logging::LogLevel;
using boost::filesystem::path;
using cv::Mat;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::make_shared;

int main(int argc, char *argv[])
{
	#ifdef WIN32
	//_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF ); // dump leaks at return
	//_CrtSetBreakAlloc(287);
	#endif
	
	string verboseLevelConsole;
	path sigsetFile, csvInputFile, inputDirectory;
	path outputFile;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"produce help message")
			("verbose,v", po::value<string>(&verboseLevelConsole)->implicit_value("DEBUG")->default_value("INFO","show messages with INFO loglevel or below."),
				"specify the verbosity of the console output: PANIC, ERROR, WARN, INFO, DEBUG or TRACE")
			("sigset,s", po::value<path>(&sigsetFile)->required(),
				"PaSC video XML sigset of the frames to be cropped")
			("input,i", po::value<path>(&inputDirectory)->required(),
				"directory containing the frames. Files should be in the format 'videoname.012.png'")
			("matrix,m", po::value<path>(&csvInputFile)->required(),
				"input matrix in CSV format (from Matlab), with same order as the images in the directory")
			("output,o", po::value<path>(&outputFile)->default_value("."),
				"filename of the output CSV matrix")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm); // style(po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
		if (vm.count("help")) {
			cout << "Usage: csv-framesfolder-to-pasc-csv [options]" << endl;
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
	
	Loggers->getLogger("facerecognition").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Loggers->getLogger("app").addAppender(make_shared<logging::ConsoleAppender>(logLevel));
	Logger appLogger = Loggers->getLogger("app");

	appLogger.debug("Verbose level for console output: " + logging::logLevelToString(logLevel));

	if (!fs::exists(inputDirectory) || fs::is_regular_file(inputDirectory)) {
		return EXIT_FAILURE;
	}
	
	// Read the XML sigset:
	auto sigset = facerecognition::utils::readPascSigset(sigsetFile, true);
	
	// Read the files in the directory:
	vector<path> files;
	try	{
		std::copy(fs::directory_iterator(inputDirectory), fs::directory_iterator(), std::back_inserter(files));
	}
	catch (const fs::filesystem_error& ex)
	{
		appLogger.error(ex.what());
		return EXIT_FAILURE;
	}

	std::sort(begin(files), end(files)); // Sort files alphabetically, same as in the loaded CSV - so their indices will match
	size_t indexInFilelist = 0;
	// Convert to a map - otherwise, the search will be be very slow.
	std::map<path, size_t> filesMap;
	for (auto& f : files) {
		filesMap.emplace(std::make_pair(f.stem().stem(), indexInFilelist));
		indexInFilelist++;
	}

	int numPngFiles = files.size();
	cv::Mat inputSimilarityMatrix(numPngFiles, numPngFiles, CV_32FC1);
	// Fill the inputSimilarityMatrix with the values from the CSV:
	{
		std::ifstream inputCsv(csvInputFile.string());
		int linesProcessed = 0;
		for (std::string line; std::getline(inputCsv, line);) {
			if (linesProcessed >= numPngFiles) {
				throw std::runtime_error("(linesProcessed >= numPngFiles), " + std::to_string(linesProcessed) + " >= " + std::to_string(numPngFiles));
			}
			vector<string> rowValues;
			boost::split(rowValues, line, boost::is_any_of(","), boost::token_compress_on);
			if (rowValues.size() != numPngFiles) {
				throw std::runtime_error("(rowValues.size() != numPngFiles), " + std::to_string(rowValues.size()) + " != " + std::to_string(numPngFiles));
			}
			for (auto col = 0; col < rowValues.size(); ++col) {
				inputSimilarityMatrix.at<float>(linesProcessed, col) = boost::lexical_cast<float>(rowValues[col]);
			}
			++linesProcessed;
		}
		if (linesProcessed != numPngFiles) {
			throw std::runtime_error("(linesProcessed != numPngFiles), " + std::to_string(linesProcessed) + " != " + std::to_string(numPngFiles));
		}
	}

	// To convert the dissimilarity-scores to similarity-scores later:
	double minScore, maxScore;
	cv::minMaxLoc(inputSimilarityMatrix, &minScore, &maxScore);

	// Loop over every gallery (rows), then over each target (cols)
	Mat fullSimilarityMatrix(sigset.size(), sigset.size(), CV_32FC1);
	for (auto q = 0; q < sigset.size(); ++q) {
		path queryImageName = sigset[q].dataPath.stem();
		appLogger.debug("Writing row of query subject " + queryImageName.string());
		//auto queryIter = std::find_if(begin(files), end(files), [queryImageName](const path& p) { return (p.stem().stem() == queryImageName); });
		auto queryIter = filesMap.find(queryImageName);
		if (queryIter == end(filesMap)) {
			// File not found, means somewhere we had a FTE. We can set the whole row's scores to 0.
			fullSimilarityMatrix(cv::Range(q, q + 1), cv::Range(0, fullSimilarityMatrix.cols)) = cv::Scalar::all(0.0f); // rowRange, colRange. First is inclusive, second is exclusive.
			continue; // Skip the whole row, start with the next query image
		}
		// Otherwise, loop through all targets (columns) and set the score if the target image is found as well:
		//auto queryIndex = std::distance(begin(files), queryIter);
		auto queryIndex = queryIter->second;
		for (auto t = 0; t < sigset.size(); ++t) {
			path targetImageName = sigset[t].dataPath.stem();
			//auto targetIter = std::find_if(begin(files), end(files), [targetImageName](const path& p) { return (p.stem().stem() == targetImageName); });
			auto targetIter = filesMap.find(targetImageName);
			if (targetIter == end(filesMap)) {
				// File not found, means somewhere we had a FTE. We could set the whole column to 0.0f but we loop over it on the next row anyway so it doesn't matter if we do.
				fullSimilarityMatrix.at<float>(q, t) = 0.0f;
			}
			else {
				//auto targetIndex = std::distance(begin(files), targetIter);
				auto targetIndex = targetIter->second;
				//fullSimilarityMatrix.at<float>(q, t) = inputSimilarityMatrix.at<float>(queryIndex, targetIndex);
				// We got dissimilarity-scores in the Matlab CSV, so flip around:
				fullSimilarityMatrix.at<float>(q, t) = maxScore - inputSimilarityMatrix.at<float>(queryIndex, targetIndex);
			}
		}
	}
	appLogger.info("Finished filling the full similarity matrix. Saving as CSV...");
	// Save the full matrix as CSV:
	facerecognition::utils::saveSimilarityMatrixAsCSV(fullSimilarityMatrix, outputFile);
	appLogger.info("Successfully saved PaSC CSV similarity matrix.");

	return EXIT_SUCCESS;
}
