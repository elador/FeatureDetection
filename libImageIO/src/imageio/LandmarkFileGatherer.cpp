/*
 * LandmarkFileGatherer.cpp
 *
 *  Created on: 29.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/LandmarkFileGatherer.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageio/ImageSource.hpp"
#include "imageio/LandmarkFormatParser.hpp"
#include "logging/LoggerFactory.hpp"
#include "boost/algorithm/string.hpp"
#include <stdexcept>
#include <utility>

using logging::Logger;
using logging::LoggerFactory;
using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;
using std::copy;
using std::sort;
using std::runtime_error;

namespace imageio {

LandmarkFileGatherer::LandmarkFileGatherer()
{
}

LandmarkFileGatherer::~LandmarkFileGatherer()
{
}

vector<path> LandmarkFileGatherer::gather(const shared_ptr<const ImageSource> imageSource, const string fileExtension, const GatherMethod gatherMethod, const vector<path> additionalPaths)
{
	Logger logger = Loggers->getLogger("imageio");

	vector<path> paths;

	if (gatherMethod == GatherMethod::ONE_FILE_PER_IMAGE_SAME_DIR)
	{
		vector<path> imagePaths = imageSource->getNames();
		for (const auto& currentImagePath : imagePaths) {
			path landmarkFile = currentImagePath;
			landmarkFile.replace_extension(fileExtension);
			if (!is_regular_file(landmarkFile)) {
				logger.warn("Could not find a landmark-file for image: " + currentImagePath.string() + ". Was looking for: " + landmarkFile.string());
				continue;
			}
			paths.push_back(landmarkFile);
		}
	}
	else if (gatherMethod == GatherMethod::ONE_FILE_PER_IMAGE_DIFFERENT_DIRS)
	{
		if (additionalPaths.empty()) {
			logger.warn("GatherMethod::ONE_FILE_PER_IMAGE_DIFFERENT_DIRS was specified, but no additional paths were provided. Using GatherMethod::ONE_FILE_PER_IMAGE_SAME_DIR.");
			return gather(imageSource, fileExtension, GatherMethod::ONE_FILE_PER_IMAGE_SAME_DIR);
		}
		vector<path> imagePaths = imageSource->getNames();
		for (const auto& currentImagePath : imagePaths) {
			
			bool landmarkFileFound = false;
			// 1. Look in the directory where the image is.
			path landmarkFile = currentImagePath;
			landmarkFile.replace_extension(fileExtension);
			if (is_regular_file(landmarkFile)) {
				paths.push_back(landmarkFile);
				landmarkFileFound = true;
			}

			// 2. Look in all directories specified in additionalPaths.
			if(!landmarkFileFound) {
				path imageBasename = currentImagePath.stem();
				for (const auto& additionalPath : additionalPaths) {
					landmarkFile = additionalPath;
					landmarkFile /= imageBasename;
					landmarkFile.replace_extension(path(fileExtension));
					if (is_regular_file(landmarkFile)) {
						paths.push_back(landmarkFile);
						landmarkFileFound = true;
						break;
					}				
				}
			}

			if(!landmarkFileFound) {
				logger.warn("Could not find a landmark-file for image: " + currentImagePath.string() + " in the image directory as well as in any of the specified directories.");
			}
		}
	}
	else if (gatherMethod == GatherMethod::SEPARATE_FILES)
	{
		if (additionalPaths.empty()) {
			logger.warn("GatherMethod::SEPARATE_FILES was specified, but no additional files were provided. Unable to find any landmarks files.");
			return paths;
		}
		for (const auto& additionalPath : additionalPaths) {
			if (is_regular_file(additionalPath)) {
				paths.push_back(additionalPath);
			} else {
				logger.warn("Filename for landmarks provided, but could not find file: " + additionalPath.string());
			}
		}
		if (paths.empty()) {
			logger.warn("Filenames for gathering landmarks were provided, but could not find any of the files.");
		}
	}

	return paths;
}

} /* namespace imageio */
