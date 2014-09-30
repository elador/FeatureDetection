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
#include "boost/property_tree/xml_parser.hpp"

//using logging::Logger;
using logging::LoggerFactory;
using boost::property_tree::ptree;
using boost::filesystem::path;
using std::vector;
using std::string;

namespace facerecognition {
	namespace utils {

std::vector<FaceRecord> readPascSigset(boost::filesystem::path filename)
{
	ptree sigsetXml;
	try {
		boost::property_tree::read_xml(filename.string(), sigsetXml);
	}
	catch (boost::property_tree::xml_parser_error& e) {
		string errorMessage{ string("Error reading the sigset file: ") + e.what() };
		Loggers->getLogger("facerecognition").error(errorMessage);
		throw std::runtime_error(errorMessage);
	}

	ptree signatures = sigsetXml.get_child("biometric-signature-set");
	vector<FaceRecord> records;
	for (const auto& e : signatures) {
		if (e.first != "<xmlattr>") {
			// This is the unique subject ID:
			string bio_sig_name = e.second.get<string>("<xmlattr>.name"); // <biometric-signature name="nd1S06262">
			// Inside this tag, we have a: <presentation name="nd1R668195" modality="face-video" file-name="06262d214.mp4" file-format="mp4" />
			ptree bio_sig_presentation = e.second.get_child("presentation");
			string pres_name = bio_sig_presentation.get<string>("<xmlattr>.name"); // Unique name of the image/video ("representation")
			//string pres_modality = bio_sig_presentation.get<string>("<xmlattr>.modality");
			string pres_filename = bio_sig_presentation.get<string>("<xmlattr>.file-name");
			//string pres_fileformat = bio_sig_presentation.get<string>("<xmlattr>.file-format");

			facerecognition::FaceRecord faceRecord;
			faceRecord.identifier = pres_name;
			faceRecord.subjectId = bio_sig_name;
			faceRecord.dataPath = pres_filename;
			//faceRecord.other = pres_name; // Use this or the filename as identifier?
			records.push_back(faceRecord);
		}
	}
	return records;
}

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
		throw std::runtime_error(errorMessage);
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
