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
		ptree images = sigset.get_child("images");
		for (auto&& entry : images) {
			FaceRecord record;
			record.identifier = entry.second.get_value<string>(); // the unique identifier for this sigset entry
			record.subjectId = entry.second.get<string>("subjectId");
			record.imagePath = entry.second.get<path>("imagePath");
			record.roll = entry.second.get_optional<float>("roll");
			record.pitch = entry.second.get_optional<float>("pitch");
			record.yaw = entry.second.get_optional<float>("yaw");
			record.session = entry.second.get<string>("session", "");
			record.lighting = entry.second.get<string>("lighting", "");
			record.expression = entry.second.get<string>("expression", "");
			record.other = entry.second.get<string>("other", "");
			faceRecords.push_back(record);
		}
	}
	catch (const boost::property_tree::ptree_error& error) {
		string errorMessage{ string("Error parsing the sigset file: ") + error.what() };
		Loggers->getLogger("facerecognition").error(errorMessage);
		throw error;
	}

	return faceRecords;
}

	} /* namespace utils */
} /* namespace facerecognition */
