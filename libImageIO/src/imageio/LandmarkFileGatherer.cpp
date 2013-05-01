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
#include "boost/algorithm/string.hpp"
#include <stdexcept>
#include <utility>

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

vector<path> LandmarkFileGatherer::gather(shared_ptr<ImageSource> imageSource, string fileExtension, GatherMethod gatherMethod)
{

	vector<path> paths;
	
/*		int endingIdx = filepath.string().find_last_of(".");
		string pathAndBasename = filepath.string().substr(0, endingIdx);
		string lmFilename = pathAndBasename + ".did";*/


	return paths;
}

} /* namespace imageio */
