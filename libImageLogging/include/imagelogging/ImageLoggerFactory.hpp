/*
 * ImageLoggerFactory.hpp
 *
 *  Created on: 04.06.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef IMAGELOGGERFACTORY_HPP_
#define IMAGELOGGERFACTORY_HPP_

#include "imagelogging/ImageLogger.hpp" // We wouldn't need this include, we could use a forward declaration. But this way, our libs/apps only have to include one file to use the logger.
#include "imagelogging/imageloglevels.hpp" // Same as for ImageLogger.hpp.

#include <map>
#include <string>
#include <memory>

using std::map;
using std::string;
using std::make_shared;

namespace imagelogging {

#define ImageLoggers ImageLoggerFactory::Instance()

/**
 * Logger factory that manages and exposes different loggers to the user.
 */
class ImageLoggerFactory
{
private:
	/* Private constructor, destructor and copy constructor - we only want one instance of the factory. */
	ImageLoggerFactory();
	~ImageLoggerFactory();
	ImageLoggerFactory(const ImageLoggerFactory &);
	ImageLoggerFactory& operator=(const ImageLoggerFactory &);

public:
	static ImageLoggerFactory* Instance();

	/**
	 * Returns the specified logger. If it is not found, creates a new logger that logs nothing.
	 *
	 * @param[in] name The name of the logger.
	 * @return The specified logger or a new one that logs nothing, if not yet created.
	 */
	ImageLogger& getLogger(const string name);

	/**
	 * Extracts the file basename and returns the logger for the file or a new logger with no logging if it hasn't been created yet.
	 *
	 * @param[in] name The file name or full file path.
	 * @return The specified logger or a new one if not yet created.
	 */
	ImageLogger& getLoggerFor(const string fileName);

	/**
	 * Loads all the loggers and their configuration from a config file. TODO!
	 *
	 * @param[in] filename The name of the config file to load.
	 * @return Maybe something
	 */
	void load(const string filename) {}; // Make static?

private:
	map<string, ImageLogger> loggers;	///< A map of all the loggers and their names.
};

} /* namespace imagelogging */
#endif /* IMAGELOGGERFACTORY_HPP_ */
