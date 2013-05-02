/*
 * LandmarkSource.cpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/LandmarkSource.hpp"
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

LandmarkSource::LandmarkSource(vector<path> landmarkFiles, shared_ptr<LandmarkFormatParser> fileParser) {
	 for (const auto& file : landmarkFiles) {
		 map<path, LandmarkCollection> lms = fileParser->read(file);
		 landmarkCollections.insert(begin(lms), end(lms));
	 }
}

LandmarkSource::~LandmarkSource() {}

const LandmarkCollection LandmarkSource::get(path imagePath) {
	// Todo: .at throws an out_of_range exception when the element is not found.
	//       [] inserts a new, empty element if not found. Think about what we want.
	return landmarkCollections.at(imagePath);
}

} /* namespace imageio */
