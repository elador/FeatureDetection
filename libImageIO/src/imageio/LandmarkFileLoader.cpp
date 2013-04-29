/*
 * LandmarkFileLoader.cpp
 *
 *  Created on: 29.04.2013
 *      Author: Patrik Huber
 */

#include "imageio/LandmarkFileLoader.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "imageio/FilebasedImageSource.hpp"
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

LandmarkFileLoader::LandmarkFileLoader()
{

}

LandmarkFileLoader::~LandmarkFileLoader()
{

}



map<path, LandmarkCollection> LandmarkFileLoader::loadOnePerImage(shared_ptr<FilebasedImageSource> imageSource, shared_ptr<LandmarkFormatParser> landmarkFormatParser)
{
	// Note: What if the landmarks are in another dir... :-)
	map<path, LandmarkCollection> lmcollMap;
	// loop over all images in imageSource
	for (const auto& filepath : imageSource->getPaths()) {
		// we have the image filepath now, how do we get to the image? E.g. what's the ending of the filename we're looking for? Depends on the landmarkFormatParser...
		vector<string> lines;
		// BIG TODO for now: replace ending (after .) by .did
		int endingIdx = filepath.string().find_last_of(".");
		string pathAndBasename = filepath.string().substr(0, endingIdx);
		string lmFilename = pathAndBasename + ".did";
		std::ifstream ifLM(lmFilename);	// Todo use the boost function that can handle a 'path'
		string strLine;
		while(getline(ifLM, strLine))
		{
			boost::algorithm::trim(strLine);
			if (!strLine.empty())
			{
				lines.push_back(strLine);
			}
		}
		LandmarkCollection lmc = landmarkFormatParser->read(lines);	//
		lmcollMap.insert(std::make_pair(filepath, lmc));
	}
	// parse with landmarkFormatParser
	return lmcollMap;
}

} /* namespace imageio */
