/*
 * ConfigParser.hpp
 *
 *  Created on: 02.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef CONFIGPARSER_HPP_
#define CONFIGPARSER_HPP_

#include <string>
#include <memory>
#include <map>
#include <vector>

using std::string;
using std::shared_ptr;
using std::map;
using std::vector;

namespace classification {
	class Classifier;
}
using classification::Classifier;

namespace detection {
	class Detector;
}
using detection::Detector;

namespace configparser {

struct ConfigLine
{
	string command;
	string variable;
	string className;
	vector<string> classArguments;
};

/**
 * Logger factory that manages and exposes different loggers to the user.
 */
class ConfigParser
{
public:
	ConfigParser();

	~ConfigParser();

	/**
	 * Returns the specified logger. If it is not found, creates a new logger that logs to the console.
	 *
	 * @param[in] name The name of the logger.
	 * @return The specified logger or a new one that logs to the console if not yet created.
	 */

	void parse(const string configFilename);

	ConfigLine parseLine(const string line);

	map<string, Classifier> readClassifiers(const string configFilename);
	static shared_ptr<Classifier> readClassifier(const string configFilename);	// move to SimpleParser?
	map<string, Detector> readDetectors(const string configFilename);
	static shared_ptr<Detector> readDetector(const string configFilename);

};

} /* namespace configparser */
#endif /* CONFIGPARSER_HPP_ */
