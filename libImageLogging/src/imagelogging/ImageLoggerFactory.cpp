/*
 * ImageLoggerFactory.cpp
 *
 *  Created on: 04.06.2013
 *      Author: Patrik Huber
 */

#include "imagelogging/ImageLoggerFactory.hpp"
#include <utility>
#include <sstream>

using std::pair;
using std::ostringstream;

namespace imagelogging {

ImageLoggerFactory::ImageLoggerFactory()
{
}


ImageLoggerFactory::~ImageLoggerFactory()
{
}

ImageLoggerFactory* ImageLoggerFactory::Instance()
{
	static ImageLoggerFactory instance;
	return &instance;
}

ImageLogger& ImageLoggerFactory::getLogger(const string name)
{
	map<string, ImageLogger>::iterator it = loggers.find(name);
	if (it != loggers.end()) {
		return it->second;	// We found the logger, return it
	} else {
		ImageLogger logger = ImageLogger(name);	// Create a new logger with no appenders.
		//logger.addAppender(make_shared<ConsoleAppender>());
		pair<map<string, ImageLogger>::iterator, bool> ret = loggers.insert(make_pair(name, logger));
		if (ret.second == false) {
			// Key already existed, although we just checked? Should be impossible to reach here.
			// Throw something? Log?
		}
		return ret.first->second;
	}
	// Note: For an empty logger with default constructor, we could just do this here: return loggers[name];
}

ImageLogger& ImageLoggerFactory::getLoggerFor(const string fileName)
{
	int lastSlash = fileName.find_last_of("/\\");
	int lastDot = fileName.find_last_of(".");
	return getLogger(fileName.substr(lastSlash + 1, lastDot - lastSlash - 1));
}

} /* namespace imagelogging */
