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
using boost::filesystem::path;
using std::string;
using std::vector;
using std::shared_ptr;

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

	vector<path> landmarkFiles;

	if (gatherMethod == GatherMethod::ONE_FILE_PER_IMAGE_SAME_DIR)
	{
		vector<path> imagePaths = imageSource->getNames();
		for (const auto& currentImagePath : imagePaths) {
			path landmarkFile = currentImagePath;
			landmarkFile.replace_extension(fileExtension);
			if (!is_regular_file(landmarkFile)) {
				logger.warn("Could not find a landmark file for image: " + currentImagePath.string() + ". Was looking for: " + landmarkFile.string());
				continue;
			}
			landmarkFiles.push_back(landmarkFile);
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
				landmarkFiles.push_back(landmarkFile);
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
						landmarkFiles.push_back(landmarkFile);
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
			logger.warn("GatherMethod::SEPARATE_FILES was specified, but no additional files were provided. No landmarks will be loaded.");
			return vector<path>();
		}
		for (const auto& additionalPath : additionalPaths) {
			if (is_regular_file(additionalPath)) {
				landmarkFiles.push_back(additionalPath);
			} else {
				logger.warn("Filename for landmarks provided, but could not find file: " + additionalPath.string());
			}
		}
		if (landmarkFiles.empty()) {
			logger.warn("Filenames for gathering landmarks were provided, but could not find any of the files.");
		}
	}
	else if (gatherMethod == GatherMethod::SEPARATE_FOLDERS)
	{
		for (const auto& directory : additionalPaths) {
			if (!boost::filesystem::exists(directory)) {
				logger.warn("The following folder specified to load landmarks from does not exist: " + directory.string());
				continue;
			}
			if (!boost::filesystem::is_directory(directory)) {
				logger.warn("The following folder specified to load landmarks from is not a directory: " + directory.string());
				continue;
			}
			// It is a valid directory, go on:
			vector<path> allFiles;
			std::copy(boost::filesystem::directory_iterator(directory), boost::filesystem::directory_iterator(), std::back_inserter(allFiles));

			vector<string> landmarkExtensions = { fileExtension };
			// We've previously added all files in the directory, so now we'll remove the ones without the desired file extension
			auto newFilesEnd = std::remove_if(begin(allFiles), end(allFiles), [&](const path& file) {
				string extension = file.extension().string();
				std::transform(begin(extension), end(extension), begin(extension), ::tolower);
				return std::none_of(begin(landmarkExtensions), end(landmarkExtensions), [&](const string& imageExtension) {
					return imageExtension == extension;
				});
			});
			allFiles.erase(newFilesEnd, end(allFiles));
			// Add to the vector where we're gathering all landmark files:
			landmarkFiles.insert(end(landmarkFiles), begin(allFiles), end(allFiles));
		}
	}
	else if (gatherMethod == GatherMethod::SEPARATE_FOLDERS_RECURSIVE)
	{
		for (const auto& directory : additionalPaths) {
			if (!boost::filesystem::exists(directory)) {
				logger.warn("The following folder specified to load landmarks from does not exist: " + directory.string());
				continue;
			}
			if (!boost::filesystem::is_directory(directory)) {
				logger.warn("The following folder specified to load landmarks from is not a directory: " + directory.string());
				continue;
			}
			// It is a valid directory, go on, recursively inside all subdirectories
			vector<path> allFiles;
			for (boost::filesystem::recursive_directory_iterator end, dir(directory); dir != end; ++dir) {
				if (boost::filesystem::is_regular_file(*dir)) {
					if ((*dir).path().extension() == fileExtension) {
						allFiles.push_back((*dir).path());
					}
				}
			}
			// Add to the vector where we're gathering all landmark files:
			landmarkFiles.insert(end(landmarkFiles), begin(allFiles), end(allFiles));
		}
	}

	return landmarkFiles;
}

} /* namespace imageio */
