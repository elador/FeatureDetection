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
	index = end(landmarkCollections); // We initialize to end() to be able to allow getName() and getLandmarks() to check if the iterator has been initialized. (If it isn't, there's no way to check it!)
	iteratorIsBeforeBegin = true;
}

void DefaultNamedLandmarkSource::reset()
{
	index = end(landmarkCollections); // Same reason as given in constructor
	iteratorIsBeforeBegin = true;
}

bool DefaultNamedLandmarkSource::next()
{
	if (iteratorIsBeforeBegin) {
		index = begin(landmarkCollections);
		iteratorIsBeforeBegin = false;
	} else {
		++index;
	}
	if (index == end(landmarkCollections)) {
		return false;
	}
	return true;
}

imageio::LandmarkCollection DefaultNamedLandmarkSource::get()
{
	next();
	// TODO: Careful, this can return an invalid element (will probably throw). Is that intentional? How to solve?
	// We could do 'if return of next() == false, return LandmarkCollection() (empty)'
	// But then we kind of never know when we're finished, because it's also possible that there are no
	// landmarks for one image, but the next one is fine again.
	// BobotLandmarkSource just seems to return an empty LandmarkCollection.
	return index->second;
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
			logger.warn("Landmarks for the given image could not be found, returning an empty LandmarkCollection. Is this expected?");
		}
		// we succeeded using the basename
	}
	return landmarks; // empty if element not found
}

imageio::LandmarkCollection DefaultNamedLandmarkSource::getLandmarks() const
{
	if (index != end(landmarkCollections)) {
		return index->second;
	}
	else {
		throw std::logic_error("Cannot return the current landmarks, iterator not initialized. Either call get(const path& imagePath) or use next().");
	}
}

boost::filesystem::path DefaultNamedLandmarkSource::getName() const
{
	if (index != end(landmarkCollections)) {
		return index->first;
	}
	else {
		throw std::logic_error("Cannot return the current landmarks, iterator not initialized. Either call get(const path& imagePath) or use next().");
	}
}

} /* namespace imageio */
