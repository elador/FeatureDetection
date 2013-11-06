/*
 * DefaultNamedLandmarkSource.cpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/DefaultNamedLandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageio/LandmarkFormatParser.hpp"
#include <stdexcept>

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

DefaultNamedLandmarkSource::~DefaultNamedLandmarkSource() {}

const LandmarkCollection& DefaultNamedLandmarkSource::get(const path& imagePath) {
	// Todo: .at throws an out_of_range exception when the element is not found.
	//       [] inserts a new, empty element if not found. Think about what we want.
	//       there is also find, which returns an iterator
	//       and we may want to use an unordered_map because of O(1) access
	
	// We can't use a local variable because we return a reference (that gets invalid once we return)
	// This is all very 'hacky'. I don't know a good solution, except changing the return type to
	// 'LandmarkCollection' (without reference). This might be a good idea anyway if we want to return
	// an empty collection.
	bool fullpathFailed = false;
	bool basenameFailed = false;
	try {
		landmarkCollections.at(imagePath);
	} catch (std::out_of_range& e) {
		fullpathFailed = true;
		try {
			landmarkCollections.at(imagePath.stem());
		} catch (std::out_of_range& e) {
			basenameFailed = true;
			// Logger: Error: Could not find
			// TODO Or just warn: The LandmarkSource does not contain landmarks for the given image. Is this expected?
			// and return Empty (be careful with local var., as we return a reference)
		}
		// we succeeded using the basename
	}
	if (!fullpathFailed) {
		return landmarkCollections.at(imagePath);
	} else if (!basenameFailed) {
		return landmarkCollections.at(imagePath.stem());
	} else {
		return LandmarkCollection(); // TODO this is going to fail, see todo above!
	}

}

} /* namespace imageio */
