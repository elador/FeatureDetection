/*
 * LandmarkSource.cpp
 *
 *  Created on: 28.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/LandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageio/LandmarkFileLoader.hpp"
#include <stdexcept>

using boost::filesystem::exists;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;
using std::copy;
using std::sort;
using std::runtime_error;

namespace imageio {

LandmarkSource::LandmarkSource(map<path, LandmarkCollection> landmarkCollections) : landmarkCollections(landmarkCollections) {
	 //landmarkCollections = landmarkFileLoader->load();
}

LandmarkSource::~LandmarkSource() {}

const LandmarkCollection LandmarkSource::get(path imagePath) {
	// Todo: .at throws an out_of_range exception when the element is not found.
	//       [] inserts a new, empty element if not found. Think about what we want.
	return landmarkCollections.at(imagePath);
}

} /* namespace imageio */
