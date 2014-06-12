/*
* LandmarkMapper.hpp
*
*  Created on: 19.04.2014
*      Author: Patrik Huber
*/

#include "imageio/LandmarkMapper.hpp"
#include "logging/LoggerFactory.hpp"

#include "imageio/RectLandmark.hpp"
#include "imageio/ModelLandmark.hpp"

#include "boost/lexical_cast.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"

using boost::property_tree::ptree;
using boost::lexical_cast;
using std::string;
using std::shared_ptr;
using std::make_shared;

namespace imageio {

LandmarkMapper::LandmarkMapper()
{
}

LandmarkMapper::LandmarkMapper(boost::filesystem::path filename)
{
	logging::Logger logger = logging::Loggers->getLogger("imageio");
	
	ptree configTree;
	try {
		boost::property_tree::info_parser::read_info(filename.string(), configTree);
	}
	catch (const boost::property_tree::ptree_error& error) {
		string errorMessage = string("LandmarkMapper: Error reading landmark-mappings file: ") + error.what();
		logger.error(errorMessage);
		throw std::runtime_error(errorMessage);
	}

	try {
		ptree ptLandmarkMappings = configTree.get_child("landmarkMappings");
		for (const auto& mapping : ptLandmarkMappings) {
			landmarkMappings.insert(make_pair(mapping.first, mapping.second.get_value<string>()));
		}
		logger.info("Loaded a list of " + lexical_cast<string>(landmarkMappings.size()) + " landmark mappings.");
	}
	catch (const boost::property_tree::ptree_error& error) {
		string errorMessage = string("LandmarkMapper: Error while parsing the mappings file: ") + error.what();
		logger.error(errorMessage);
		throw std::runtime_error(errorMessage);
	}
	catch (const std::runtime_error& error) {
		string errorMessage = string("LandmarkMapper: Error while parsing the mappings file: ") + error.what();
		logger.error(errorMessage);
		throw std::runtime_error(errorMessage);
	}
}

LandmarkMapper LandmarkMapper::load(boost::filesystem::path filename)
{
	return LandmarkMapper(filename);
}

string LandmarkMapper::convert(string landmarkName)
{
	return landmarkMappings.at(landmarkName); // throws an out_of_range exception if landmarkName does not match the key of any element in the map
}


shared_ptr<Landmark> LandmarkMapper::convert(shared_ptr<Landmark> landmark)
{
	shared_ptr<Landmark> convertedLandmark;
	string mappedId = landmarkMappings.at(landmark->getName());
	switch (landmark->getType())
	{
	case Landmark::LandmarkType::MODEL:
		convertedLandmark = make_shared<ModelLandmark>(mappedId, landmark->getPosition3D(), landmark->isVisible());
		break;
	case Landmark::LandmarkType::RECT:
		convertedLandmark = make_shared<RectLandmark>(mappedId, landmark->getPosition2D(), landmark->getSize(), landmark->isVisible());
		break;
	default:
		logging::Logger logger = logging::Loggers->getLogger("imageio");
		string errorMessage = "Encountered an unknown LandmarkType. Please update this switch-statement.";
		logger.error(errorMessage);
		throw std::runtime_error(errorMessage);
		break;
	}
	return convertedLandmark;
}

LandmarkCollection LandmarkMapper::convert(LandmarkCollection landmarks)
{
	logging::Logger logger = logging::Loggers->getLogger("imageio");

	LandmarkCollection convertedLandmarks;
	const auto& originalLandmarks = landmarks.getLandmarks();
	for (const auto& lm : originalLandmarks) {
		try {
			shared_ptr<Landmark> convertedLandmark = convert(lm);
			convertedLandmarks.insert(convertedLandmark);
		}
		catch (std::out_of_range& error) {
			// a mapping for the current landmark is not found, we skip it
			logger.trace("Could not find the current landmark \"" + lm->getName() + "\" in the landmarks-mapping, not converting this landmark.");
		}
	}
	return convertedLandmarks;
}

} /* namespace imageio */
