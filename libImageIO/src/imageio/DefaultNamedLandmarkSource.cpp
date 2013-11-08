/*
 * DefaultNamedLandmarkSource.cpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageio/LandmarkFormatParser.hpp"
#include "logging/LoggerFactory.hpp"
#include <stdexcept>

using logging::Logger;
using logging::LoggerFactory;
using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;
using std::copy;
using std::sort;
using std::runtime_error;
using std::make_pair;

namespace imageio {

DefaultNamedLandmarkSource::DefaultNamedLandmarkSource(vector<path> landmarkFiles, shared_ptr<LandmarkFormatParser> fileParser) {
	 for (const auto& file : landmarkFiles) {
		 map<path, LandmarkCollection> lms = fileParser->read(file);
		 landmarkCollections.insert(begin(lms), end(lms));
	 }
}

LandmarkCollection DefaultNamedLandmarkSource::get(const path& imagePath) {
	// Todo: .at throws an out_of_range exception when the element is not found.
	//       [] inserts a new, empty element if not found. Think about what we want.
	//       there is also find, which returns an iterator
	//       and we may want to use an unordered_map because of O(1) access
	Logger logger = Loggers->getLogger("imageio");

	LandmarkCollection landmarks;
	try {
		landmarks = landmarkCollections.at(imagePath);
	} catch (std::out_of_range& e) {
		try {
			landmarks = landmarkCollections.at(imagePath.stem());
		} catch (std::out_of_range& e) {
			logger.info("The LandmarkSource does not contain landmarks for the given image. Is this expected?");
		}
		// we succeeded using the basename
	}
	return landmarks;
}

} /* namespace imageio */
