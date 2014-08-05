/*
 * DidLandmarkSink.cpp
 *
 *  Created on: 05.04.2014
 *      Author: Patrik Huber
 */

#include "imageio/DidLandmarkSink.hpp"
#include "imageio/Landmark.hpp"
#include "imageio/LandmarkCollection.hpp"

#include <fstream>

using boost::filesystem::path;

namespace imageio {

void DidLandmarkSink::add(const LandmarkCollection& collection, path filename)
{
	filename.replace_extension(".did");
	std::ofstream outputFile(filename.string());

	if (!outputFile.is_open()) {
		// TODO log
		throw std::runtime_error("DidLandmarkSink: Error creating the output file " + filename.string());
	}

	const auto& landmarks = collection.getLandmarks();
	for (const auto& lm : landmarks) {
		if (lm->isVisible()) {
			// The .did files cannot handle visibility, a landmark is always used.
			// Thus, we only write the landmark if it is visible.
			outputFile << lm->getX() << " " << lm->getY() << " " << lm->getName() << std::endl;
		}
	}

	outputFile.close();
	return;
}

} /* namespace imageio */
