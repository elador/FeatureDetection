/*
 * pasc.cpp
 *
 *  Created on: 27.09.2014
 *      Author: Patrik Huber
 */
#include "facerecognition/pasc.hpp"

#include "logging/LoggerFactory.hpp"

#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"

#include <fstream>

using logging::LoggerFactory;
using boost::lexical_cast;
using std::string;
using std::vector;

namespace facerecognition {

std::string getZeroPadded(int pascFrameNumber)
{
	std::ostringstream ss;
	ss << std::setw(3) << std::setfill('0') << pascFrameNumber;
	return ss.str();
}

std::string getPascFrameName(boost::filesystem::path videoFilename, int pascFrameNumber)
{
	std::ostringstream ss;
	ss << std::setw(3) << std::setfill('0') << pascFrameNumber;
	return videoFilename.stem().string() + "/" + videoFilename.stem().string() + "-" + ss.str() + ".jpg";
}

std::vector<PascVideoDetection> readPascVideoDetections(boost::filesystem::path csvFile)
{
	std::vector<PascVideoDetection> detections;

	std::ifstream file(csvFile.string());
	if (!file.is_open() || !file.good()) {
		throw std::runtime_error("Error while trying to open and read the file.");
	}
	string line;
	std::getline(file, line); // The header line, we skip it

	while (std::getline(file, line))
	{
		PascVideoDetection detection;

		vector<string> tokens;
		boost::trim_right_if(line, boost::is_any_of("\r")); // Windows line-endings are \r\n, Linux only \n. Thus, when a file has been created on windows and is read on linux, we need to remove the trailing \r.
		boost::split(tokens, line, boost::is_any_of(","));

		detection.frame_id = tokens[0];
		detection.fcen_x = lexical_cast<int>(tokens[1]);
		detection.fcen_y = lexical_cast<int>(tokens[2]);
		detection.fwidth = lexical_cast<int>(tokens[3]);
		detection.fheight = lexical_cast<int>(tokens[4]);
		detection.fpose_y = lexical_cast<float>(tokens[5]);
		if (tokens[6] != "") {
			detection.le_x = lexical_cast<int>(tokens[6]);
		}
		if (tokens[7] != "") {
			detection.le_y = lexical_cast<int>(tokens[7]);
		}
		if (tokens[8] != "") {
			detection.re_x = lexical_cast<int>(tokens[8]);
		}
		if (tokens[9] != "") {
			detection.re_y = lexical_cast<int>(tokens[9]);
		}
		detections.emplace_back(detection);
	}
	return detections;
}

} /* namespace facerecognition */
