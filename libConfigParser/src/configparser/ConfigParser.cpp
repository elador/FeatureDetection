/*
 * ConfigParser.cpp
 *
 *  Created on: 02.03.2013
 *      Author: Patrik Huber
 */

#include "configparser/ConfigParser.hpp"
#include "boost/algorithm/string.hpp"
#include <fstream>
#include <algorithm>

namespace configparser {

ConfigParser::ConfigParser()
{
}


ConfigParser::~ConfigParser()
{
}

void ConfigParser::parse(const string configFilename)
{
	std::ifstream file;
	file.open(configFilename, std::ios::in);
	if (!file.is_open()) {
		throw std::ios_base::failure("Error: Could not open config file: " + configFilename);
	}
	std::string line;
	while (std::getline(file, line)) {
		parseLine(line);
	}

	file.close();
}

ConfigLine ConfigParser::parseLine(const string line)
{
	ConfigLine parsedLine;

	vector<string> commenting;
	boost::split(commenting, line, boost::is_any_of("#"));
	string realLine;
	if(commenting.size()>1)
		realLine = commenting[0];
	else
		realLine = line;

	boost::trim(realLine);
	if(realLine.empty())
		return ConfigLine();

	vector<string> colonSplit;
	boost::split(colonSplit, realLine, boost::is_any_of(":"));

	vector<string> commandSplit;
	boost::split(commandSplit, colonSplit[0], boost::is_any_of(" "));
	string cmd;
	string varName;
	if (commandSplit.size()>1) {
		parsedLine.command = commandSplit[0];
		parsedLine.variable = commandSplit[1];
	} else {
		parsedLine.command = commandSplit[0];
		parsedLine.variable = "";
	}

	unsigned int bracketsStart = colonSplit[1].find("(");
	unsigned int bracketsEnd = colonSplit[1].find(")");	// TODO check for string::npos (=no matches found)
	
	string className = colonSplit[1].substr(0, bracketsStart);
	boost::trim(className);
	parsedLine.className = className;

	string arguments = colonSplit[1].substr(bracketsStart+1, bracketsEnd-bracketsStart-1); // TODO What happens when nothing in bracket?
	//arguments.erase(std::remove_if(arguments.begin(), arguments.end(), ' '), arguments.end());
	boost::erase_all(arguments, " ");
	
	vector<string> argumentsCommaSplit;
	boost::split(argumentsCommaSplit, arguments, boost::is_any_of(","));

	vector<string> finalArguments;
	for (auto arg : argumentsCommaSplit) {
		vector<string> tmp;
		boost::split(tmp, arg, boost::is_any_of("="));
		if(tmp.size()>1) {
			finalArguments.push_back(tmp[1]);
		} else {
			finalArguments.push_back(tmp[0]);
		}
	}
	parsedLine.classArguments = finalArguments;

	return parsedLine;
}

} /* namespace configparser */
