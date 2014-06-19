/*
 * utils.cpp
 *
 *  Created on: 15.06.2014
 *      Author: Patrik Huber
 */
#include "facerecognition/utils.hpp"

#include "logging/LoggerFactory.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"

//using logging::Logger;
using logging::LoggerFactory;
using boost::property_tree::ptree;
using boost::filesystem::path;
using std::vector;
using std::string;

namespace facerecognition {
	namespace utils {

std::vector<FaceRecord> readSigset(boost::filesystem::path filename)
{
	vector<FaceRecord> faceRecords;
	ptree sigset;
	try {
		read_info(filename.string(), sigset);
	}
	catch (const boost::property_tree::ptree_error& error) {
		string errorMessage{ string("Error reading the sigset file: ") + error.what() };
		Loggers->getLogger("facerecognition").error(errorMessage);
		throw error;
	}
	try {
		ptree records = sigset.get_child("records");
		for (auto&& entry : records) {
			FaceRecord faceRecord;
			faceRecord.identifier = entry.second.get_value<string>(); // the unique identifier for this sigset entry
			faceRecord.subjectId = entry.second.get<string>("subjectId");
			faceRecord.dataPath = entry.second.get<path>("dataPath");
			faceRecord.roll = entry.second.get_optional<float>("roll");
			faceRecord.pitch = entry.second.get_optional<float>("pitch");
			faceRecord.yaw = entry.second.get_optional<float>("yaw");
			faceRecord.session = entry.second.get<string>("session", "");
			faceRecord.lighting = entry.second.get<string>("lighting", "");
			faceRecord.expression = entry.second.get<string>("expression", "");
			faceRecord.other = entry.second.get<string>("other", "");
			faceRecords.push_back(faceRecord);
		}
	}
	catch (const boost::property_tree::ptree_error& error) {
		string errorMessage{ string("Error parsing the sigset file: ") + error.what() };
		Loggers->getLogger("facerecognition").error(errorMessage);
		throw error;
	}

	return faceRecords;
}

path transformDataPath(const path& originalDataPath, DataPathTransformation transformation)
{
	path dataPath;
	if (transformation.name == "original") {
		dataPath = originalDataPath;
	}
	else if (transformation.name == "basename") {
		dataPath = originalDataPath.filename();
	}
	string basename = dataPath.stem().string();
	string originalExtension = dataPath.extension().string();
	if (!transformation.prefix.empty()) {
		basename = transformation.prefix + basename;
	}
	if (!transformation.suffix.empty()) {
		basename = basename + transformation.suffix;
	}
	path filename(basename + originalExtension);
	if (!transformation.replaceExtension.empty()) {
		filename.replace_extension(transformation.replaceExtension);
	}
	path fullFilePath;
	if (transformation.name == "original") {
		auto lastSlash = dataPath.string().find_last_of("/\\");
		fullFilePath = transformation.rootPath / dataPath.string().substr(0, lastSlash) / filename;
	}
	else if (transformation.name == "basename") {
		fullFilePath = transformation.rootPath / filename;
	}
	return fullFilePath;
}

	} /* namespace utils */
} /* namespace facerecognition */
