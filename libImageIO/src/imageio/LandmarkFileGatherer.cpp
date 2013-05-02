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

vector<path> LandmarkFileGatherer::gather(shared_ptr<ImageSource> imageSource, string fileExtension, GatherMethod gatherMethod, vector<path> additionalPaths)
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
		if (additionalPaths.size() == 0) {
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

	}
	//path basename = currentImagePath.stem();
	//path directory = currentImagePath.parent_path();

	
	
/*		int endingIdx = filepath.string().find_last_of(".");
		string pathAndBasename = filepath.string().substr(0, endingIdx);
		string lmFilename = pathAndBasename + ".did";*/


	return paths;
}

} /* namespace imageio */
